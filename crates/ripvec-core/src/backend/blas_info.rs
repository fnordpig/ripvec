//! Runtime BLAS detection and optimization recommendations.
//!
//! Probes the linked BLAS library at runtime via `dlsym` and recommends
//! the optimal BLAS for the current CPU vendor.

/// Detected BLAS library.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlasKind {
    /// `OpenBLAS` (good default, not optimal on AMD or Intel).
    OpenBlas,
    /// Intel MKL / oneMKL (optimal on Intel, crippled on AMD).
    IntelMkl,
    /// BLIS or AMD AOCL-BLAS (optimal on AMD).
    Blis,
    /// Apple Accelerate (optimal on Apple Silicon).
    Accelerate,
    /// Unknown or no external BLAS (pure Rust ndarray fallback).
    Unknown,
}

/// Detected CPU vendor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuVendor {
    /// AMD (Ryzen, EPYC, Threadripper).
    Amd,
    /// Intel (Core, Xeon).
    Intel,
    /// Apple Silicon (M-series).
    Apple,
    /// Unknown vendor.
    Unknown,
}

impl std::fmt::Display for BlasKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OpenBlas => write!(f, "OpenBLAS"),
            Self::IntelMkl => write!(f, "Intel MKL"),
            Self::Blis => write!(f, "BLIS/AOCL"),
            Self::Accelerate => write!(f, "Apple Accelerate"),
            Self::Unknown => write!(f, "pure Rust (no external BLAS)"),
        }
    }
}

impl std::fmt::Display for CpuVendor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Amd => write!(f, "AMD"),
            Self::Intel => write!(f, "Intel"),
            Self::Apple => write!(f, "Apple"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

/// Detect the CPU vendor from `/proc/cpuinfo` (Linux) or sysctl (macOS).
#[must_use]
pub fn detect_cpu_vendor() -> CpuVendor {
    #[cfg(target_os = "macos")]
    {
        return CpuVendor::Apple;
    }

    #[cfg(target_os = "linux")]
    {
        if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
            if cpuinfo.contains("AuthenticAMD") {
                return CpuVendor::Amd;
            }
            if cpuinfo.contains("GenuineIntel") {
                return CpuVendor::Intel;
            }
        }
    }

    CpuVendor::Unknown
}

/// Detect which BLAS library is linked at runtime.
///
/// Probes for vendor-specific symbols using `dlsym(RTLD_DEFAULT, ...)`.
#[must_use]
pub fn detect_blas() -> BlasKind {
    #[cfg(target_os = "macos")]
    {
        // On macOS, ndarray links Accelerate by default
        return BlasKind::Accelerate;
    }

    #[cfg(target_os = "linux")]
    {
        // Probe for vendor-specific symbols in the loaded libraries
        use std::ffi::CString;

        let probe = |symbol: &str| -> bool {
            let c_sym = CString::new(symbol).unwrap();
            #[expect(unsafe_code, reason = "dlsym probe for BLAS detection")]
            unsafe {
                !libc::dlsym(libc::RTLD_DEFAULT, c_sym.as_ptr()).is_null()
            }
        };

        // BLIS / AOCL-BLAS
        if probe("bli_info_get_version_str") {
            return BlasKind::Blis;
        }

        // Intel MKL
        if probe("mkl_get_version") {
            return BlasKind::IntelMkl;
        }

        // OpenBLAS
        if probe("openblas_get_config") {
            return BlasKind::OpenBlas;
        }

        BlasKind::Unknown
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        BlasKind::Unknown
    }
}

/// Return a recommendation string if the user isn't using the optimal BLAS.
///
/// Returns `None` if the current BLAS is optimal for the CPU, or if
/// we can't determine a better option.
#[must_use]
pub fn recommend_blas() -> Option<String> {
    let blas = detect_blas();
    let cpu = detect_cpu_vendor();

    match (cpu, blas) {
        // Already optimal
        (CpuVendor::Amd, BlasKind::Blis) => None,
        (CpuVendor::Intel, BlasKind::IntelMkl) => None,
        (CpuVendor::Apple, BlasKind::Accelerate) => None,

        // Suboptimal — recommend better
        (CpuVendor::Amd, BlasKind::OpenBlas) => Some(
            "tip: AOCL-BLAS is 10-15% faster than OpenBLAS on AMD CPUs. \
             Install: https://developer.amd.com/amd-aocl/"
                .to_string(),
        ),
        (CpuVendor::Amd, BlasKind::IntelMkl) => Some(
            "warning: Intel MKL is intentionally slow on AMD CPUs (CPUID check). \
             Use AOCL-BLAS or OpenBLAS instead."
                .to_string(),
        ),
        (CpuVendor::Intel, BlasKind::OpenBlas) => Some(
            "tip: Intel MKL is faster than OpenBLAS on Intel CPUs. \
             Install: sudo apt install libmkl-dev"
                .to_string(),
        ),

        // No BLAS at all
        (CpuVendor::Amd, BlasKind::Unknown) => Some(
            "warning: no BLAS library detected — CPU inference will be slow. \
             Install: sudo apt install libopenblas-dev (or AOCL-BLAS for best AMD performance)"
                .to_string(),
        ),
        (CpuVendor::Intel, BlasKind::Unknown) => Some(
            "warning: no BLAS library detected — CPU inference will be slow. \
             Install: sudo apt install libmkl-dev"
                .to_string(),
        ),
        (_, BlasKind::Unknown) => Some(
            "warning: no BLAS library detected — CPU inference will be slow. \
             Install: sudo apt install libopenblas-dev"
                .to_string(),
        ),

        // Everything else is fine enough
        _ => None,
    }
}
