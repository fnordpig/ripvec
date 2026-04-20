//! Compile CUDA kernels to CUBIN (SASS) directly via NVRTC.
//!
//! cudarc's `compile_ptx_with_opts` emits PTX, which the driver must then JIT
//! to SASS at `cuModuleLoadData` time. The PTX carries a version tag from the
//! CUDA Toolkit NVRTC was built from; older runtime drivers reject newer PTX
//! with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` regardless of the target compute
//! capability. We've hit this on T4G (sm_75) with a driver 590 host trying to
//! load PTX from a CUDA 13.x toolkit.
//!
//! `nvrtcGetCUBIN` (available since CUDA 11.2) has NVRTC emit SASS for a
//! concrete `sm_XX` target directly — no PTX JIT, no version gate, and
//! marginally faster startup. CUBIN compilation requires a real arch, not a
//! virtual `compute_XX` one.
//!
//! cudarc 0.19's safe wrappers only expose `get_ptx`, so we reach into
//! `cudarc::nvrtc::sys` and drive the program create/compile/get-cubin
//! lifecycle ourselves.

use std::ffi::CString;

use cudarc::nvrtc::{Ptx, sys};

/// Compile `src` to a CUBIN image for `arch` (e.g. `"sm_75"`).
///
/// The result is wrapped in `Ptx::from_binary` so callers can hand it to
/// `CudaContext::load_module` unchanged — the CUDA driver auto-detects cubin
/// vs PTX from the image header.
///
/// # Errors
///
/// Returns an error string if NVRTC fails to create the program, compile the
/// source, or extract the cubin. On compile failure the NVRTC program log is
/// included in the error.
pub fn compile_cubin(src: &str, arch: &str) -> Result<Ptx, String> {
    compile_cubin_with_extra_opts(src, arch, &[])
}

/// Variant accepting extra NVRTC options (e.g. `"--use_fast_math"`,
/// `"--maxrregcount=64"`). `--gpu-architecture=<arch>` is always prepended.
pub fn compile_cubin_with_extra_opts(
    src: &str,
    arch: &str,
    extra_opts: &[&str],
) -> Result<Ptx, String> {
    let c_src = CString::new(src).map_err(|e| format!("kernel source contains NUL: {e}"))?;

    let mut option_strings: Vec<CString> = Vec::with_capacity(1 + extra_opts.len());
    option_strings.push(
        CString::new(format!("--gpu-architecture={arch}"))
            .map_err(|e| format!("arch option contains NUL: {e}"))?,
    );
    for opt in extra_opts {
        option_strings
            .push(CString::new(*opt).map_err(|e| format!("option {opt:?} contains NUL: {e}"))?);
    }
    let option_ptrs: Vec<*const ::core::ffi::c_char> =
        option_strings.iter().map(|s| s.as_ptr()).collect();

    // SAFETY: all pointers passed to the NVRTC FFI are live for the duration
    // of each call, `prog` is driven through its full lifecycle
    // (create → compile → get-cubin → destroy) with no early return leaking
    // the handle, and the cubin size returned by `nvrtcGetCUBINSize`
    // matches the buffer we pass to `nvrtcGetCUBIN`.
    unsafe {
        let mut prog: sys::nvrtcProgram = std::ptr::null_mut();
        let r = sys::nvrtcCreateProgram(
            &mut prog,
            c_src.as_ptr(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            std::ptr::null(),
        );
        if r != sys::nvrtcResult::NVRTC_SUCCESS {
            return Err(format!("nvrtcCreateProgram failed: {r:?}"));
        }

        let compile_result = compile_and_fetch(prog, arch, &option_ptrs);
        let _ = sys::nvrtcDestroyProgram(&mut prog);
        compile_result
    }
}

/// SAFETY: `prog` must be a valid, compiled-or-uncompiled NVRTC program
/// handle. `option_ptrs` entries must remain live for the duration of the
/// call (their backing `CString`s are held by the caller).
unsafe fn compile_and_fetch(
    prog: sys::nvrtcProgram,
    arch: &str,
    option_ptrs: &[*const ::core::ffi::c_char],
) -> Result<Ptx, String> {
    // SAFETY: upheld by caller; individual FFI calls are fine because `prog`
    // is live and `option_ptrs` is a valid slice.
    let r = unsafe {
        sys::nvrtcCompileProgram(
            prog,
            option_ptrs.len() as ::core::ffi::c_int,
            option_ptrs.as_ptr(),
        )
    };
    if r != sys::nvrtcResult::NVRTC_SUCCESS {
        // SAFETY: `prog` still valid per caller contract.
        let log = unsafe { program_log(prog) }.unwrap_or_default();
        return Err(format!("nvrtcCompileProgram failed ({r:?}): {log}"));
    }

    let mut size: usize = 0;
    // SAFETY: `prog` valid; `&mut size` is a valid out-ptr.
    let r = unsafe { sys::nvrtcGetCUBINSize(prog, &mut size) };
    if r != sys::nvrtcResult::NVRTC_SUCCESS {
        return Err(format!("nvrtcGetCUBINSize failed: {r:?}"));
    }
    if size == 0 {
        return Err(format!(
            "nvrtcGetCUBIN returned empty — arch {arch:?} must be a real sm_XX target, not virtual compute_XX"
        ));
    }

    let mut buf: Vec<u8> = vec![0u8; size];
    // SAFETY: `buf` has capacity `size` that matches what `nvrtcGetCUBIN`
    // will write; pointer cast is sound because u8 and c_char share layout.
    let r = unsafe { sys::nvrtcGetCUBIN(prog, buf.as_mut_ptr() as *mut ::core::ffi::c_char) };
    if r != sys::nvrtcResult::NVRTC_SUCCESS {
        return Err(format!("nvrtcGetCUBIN failed: {r:?}"));
    }

    Ok(Ptx::from_binary(buf))
}

/// SAFETY: `prog` must be a valid NVRTC program handle.
unsafe fn program_log(prog: sys::nvrtcProgram) -> Option<String> {
    let mut size: usize = 0;
    // SAFETY: `prog` valid per caller contract; `&mut size` is a valid out-ptr.
    if unsafe { sys::nvrtcGetProgramLogSize(prog, &mut size) } != sys::nvrtcResult::NVRTC_SUCCESS {
        return None;
    }
    if size <= 1 {
        return Some(String::new());
    }
    let mut buf: Vec<u8> = vec![0u8; size];
    // SAFETY: `buf` capacity matches `size`; cast is a layout-compatible reinterpret.
    if unsafe { sys::nvrtcGetProgramLog(prog, buf.as_mut_ptr() as *mut ::core::ffi::c_char) }
        != sys::nvrtcResult::NVRTC_SUCCESS
    {
        return None;
    }
    if let Some(&0) = buf.last() {
        buf.pop();
    }
    String::from_utf8(buf).ok()
}
