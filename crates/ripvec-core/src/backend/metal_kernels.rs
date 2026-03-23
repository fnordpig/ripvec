//! Metal Shading Language (MSL) kernel sources.
//!
//! Contains MSL compute kernels used by the Metal backend. Kernels are compiled
//! at runtime via `MTLDevice::newLibraryWithSource_options_error`.

/// Simple test kernel that adds 1.0 to every element of a float buffer.
///
/// Used by integration tests to verify the full Metal pipeline: device creation,
/// MSL compilation, buffer allocation, compute dispatch, and result readback.
pub const TEST_KERNEL: &str = r"
#include <metal_stdlib>
using namespace metal;

kernel void add_one(
    device float *data [[buffer(0)]],
    uint tid [[thread_position_in_grid]]
) {
    data[tid] = data[tid] + 1.0;
}
";
