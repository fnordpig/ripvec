//! Metal GPU embedding backend for Apple Silicon.
//!
//! Provides low-level Metal device initialization, MSL kernel compilation, and
//! compute dispatch infrastructure. Uses `objc2-metal` Rust bindings to Apple's
//! Metal framework.
//!
//! # Architecture
//!
//! The Metal backend detects the GPU family at init time ([`ChipFamily`]) and
//! stores a pre-compiled compute pipeline alongside the device and command
//! queue. All GPU resources use `StorageModeShared` (unified memory on Apple
//! Silicon) to avoid explicit CPU-GPU copies.
//!
//! # Thread safety
//!
//! Metal device and command queue are thread-safe. `MetalBackend` implements
//! `Send + Sync` so it can be shared across the ring-buffer pipeline.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCommandQueue, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLGPUFamily, MTLLibrary,
};

use super::{DeviceHint, EmbedBackend, Encoding};

// ---------------------------------------------------------------------------
// CoreGraphics linkage (required for MTLCreateSystemDefaultDevice)
// ---------------------------------------------------------------------------

#[expect(unsafe_code, reason = "extern block required for CoreGraphics linkage")]
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

// ---------------------------------------------------------------------------
// Device + pipeline helpers
// ---------------------------------------------------------------------------

/// Create the default Metal GPU device.
fn create_device() -> crate::Result<Retained<ProtocolObject<dyn MTLDevice>>> {
    MTLCreateSystemDefaultDevice()
        .ok_or_else(|| crate::Error::Metal("no Metal device available".into()))
}

/// Create a command queue for submitting work.
fn create_queue(
    device: &ProtocolObject<dyn MTLDevice>,
) -> crate::Result<Retained<ProtocolObject<dyn MTLCommandQueue>>> {
    device
        .newCommandQueue()
        .ok_or_else(|| crate::Error::Metal("failed to create command queue".into()))
}

/// Compile MSL source code into a Metal library.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "will be used in BERT inference")
)]
fn compile_library(
    device: &ProtocolObject<dyn MTLDevice>,
    source: &str,
) -> crate::Result<Retained<ProtocolObject<dyn MTLLibrary>>> {
    let ns_source = NSString::from_str(source);
    device
        .newLibraryWithSource_options_error(&ns_source, None)
        .map_err(|e| crate::Error::Metal(format!("MSL compilation failed: {e}")))
}

/// Create a compute pipeline state from a named kernel function.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "will be used in BERT inference")
)]
fn create_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    name: &str,
) -> crate::Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
    let ns_name = NSString::from_str(name);
    let function = library
        .newFunctionWithName(&ns_name)
        .ok_or_else(|| crate::Error::Metal(format!("function '{name}' not found in library")))?;
    device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|e| crate::Error::Metal(format!("pipeline creation failed: {e}")))
}

// ---------------------------------------------------------------------------
// Chip family detection
// ---------------------------------------------------------------------------

/// Apple GPU family classification for tuning compute kernels.
///
/// Different chip families have different optimal GEMM tile sizes and
/// support different hardware features (e.g. async copies).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChipFamily {
    /// M1, M2 (Apple7/8) -- prefer async copy, 48x48 GEMM tiles.
    Apple7_8,
    /// M3+ (Apple9) -- no async copy, 32x32 GEMM tiles.
    Apple9,
}

impl ChipFamily {
    /// Detect the GPU family from the device capabilities.
    fn detect(device: &ProtocolObject<dyn MTLDevice>) -> Self {
        if device.supportsFamily(MTLGPUFamily::Apple9) {
            Self::Apple9
        } else {
            Self::Apple7_8
        }
    }
}

// ---------------------------------------------------------------------------
// MetalBackend
// ---------------------------------------------------------------------------

/// Metal GPU embedding backend.
///
/// Holds the device, command queue, and chip family information needed to
/// dispatch compute work on Apple Silicon GPUs. The full BERT inference
/// pipeline will be added in a follow-up; this struct currently provides
/// the initialization and kernel compilation infrastructure.
pub struct MetalBackend {
    /// The Metal GPU device.
    #[expect(
        dead_code,
        reason = "will be used for buffer allocation in BERT inference"
    )]
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    /// Command queue for submitting compute work.
    #[expect(
        dead_code,
        reason = "will be used for compute dispatch in BERT inference"
    )]
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    /// Detected GPU chip family for kernel tuning.
    #[expect(
        dead_code,
        reason = "will be used for kernel selection in BERT inference"
    )]
    chip_family: ChipFamily,
}

// SAFETY: Metal device and command queue are thread-safe (Apple documents
// MTLDevice and MTLCommandQueue as safe to use from multiple threads).
#[expect(
    unsafe_code,
    reason = "Metal device/queue are documented as thread-safe"
)]
unsafe impl Send for MetalBackend {}
// SAFETY: Same rationale — Metal's device and queue are thread-safe.
#[expect(
    unsafe_code,
    reason = "Metal device/queue are documented as thread-safe"
)]
unsafe impl Sync for MetalBackend {}

impl MetalBackend {
    /// Initialize the Metal backend: create device, queue, and detect chip family.
    ///
    /// # Errors
    ///
    /// Returns an error if no Metal device is available or the command queue
    /// cannot be created.
    pub fn load(_model_repo: &str, _device_hint: &DeviceHint) -> crate::Result<Self> {
        let device = create_device()?;
        let queue = create_queue(&device)?;
        let chip_family = ChipFamily::detect(&device);

        tracing::info!(
            device = %device.name(),
            chip_family = ?chip_family,
            "Metal backend initialized"
        );

        Ok(Self {
            device,
            queue,
            chip_family,
        })
    }
}

impl EmbedBackend for MetalBackend {
    fn embed_batch(&self, _encodings: &[Encoding]) -> crate::Result<Vec<Vec<f32>>> {
        Err(crate::Error::Metal(
            "Metal BERT inference not yet implemented".into(),
        ))
    }

    fn supports_clone(&self) -> bool {
        false
    }

    fn clone_backend(&self) -> Box<dyn EmbedBackend> {
        panic!("MetalBackend does not support cloning");
    }

    fn is_gpu(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[expect(
    unsafe_code,
    reason = "Metal buffer creation/readback requires unsafe FFI"
)]
mod tests {
    use core::ffi::c_void;
    use core::ptr::NonNull;

    use objc2_foundation::NSUInteger;
    use objc2_metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder,
        MTLResourceOptions, MTLSize,
    };

    use super::*;

    #[test]
    fn metal_device_creation() {
        let device = create_device().unwrap();
        let _queue = create_queue(&device).unwrap();
        let name = device.name().to_string();
        assert!(!name.is_empty(), "device should have a name");
    }

    #[test]
    fn metal_chip_family_detection() {
        let device = create_device().unwrap();
        let family = ChipFamily::detect(&device);
        // On any Apple Silicon Mac, should be one of the two variants.
        assert!(
            family == ChipFamily::Apple7_8 || family == ChipFamily::Apple9,
            "unexpected chip family: {family:?}"
        );
    }

    #[test]
    fn metal_kernel_compilation() {
        let device = create_device().unwrap();
        let library = compile_library(&device, super::super::metal_kernels::TEST_KERNEL).unwrap();
        let pipeline = create_pipeline(&device, &library, "add_one").unwrap();
        assert!(
            pipeline.maxTotalThreadsPerThreadgroup() > 0,
            "pipeline should support at least 1 thread per threadgroup"
        );
    }

    #[test]
    fn metal_compute_dispatch() {
        let device = create_device().unwrap();
        let queue = create_queue(&device).unwrap();
        let library = compile_library(&device, super::super::metal_kernels::TEST_KERNEL).unwrap();
        let pipeline = create_pipeline(&device, &library, "add_one").unwrap();

        // Create a buffer with [1.0, 2.0, 3.0, 4.0]
        let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let buf_size = (data.len() * core::mem::size_of::<f32>()) as NSUInteger;
        let buffer = unsafe {
            device.newBufferWithBytes_length_options(
                NonNull::new(data.as_ptr() as *mut c_void).unwrap(),
                buf_size,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .expect("failed to create Metal buffer");

        // Encode and dispatch the add_one kernel
        let cmd_buf = queue
            .commandBuffer()
            .expect("failed to create command buffer");
        let encoder = cmd_buf
            .computeCommandEncoder()
            .expect("failed to create compute encoder");
        encoder.setComputePipelineState(&pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&buffer), 0, 0);
        }

        let grid = MTLSize {
            width: 4,
            height: 1,
            depth: 1,
        };
        let threadgroup = MTLSize {
            width: 4,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreads_threadsPerThreadgroup(grid, threadgroup);
        encoder.endEncoding();

        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        // Read back results -- should be [2.0, 3.0, 4.0, 5.0]
        let result =
            unsafe { core::slice::from_raw_parts(buffer.contents().as_ptr() as *const f32, 4) };
        assert_eq!(result, &[2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn metal_backend_load() {
        let backend = MetalBackend::load("test/model", &DeviceHint::Auto).unwrap();
        assert!(backend.is_gpu());
        assert!(!backend.supports_clone());
    }
}
