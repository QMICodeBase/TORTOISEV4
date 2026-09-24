// Smoke-test kernel: proves the WGSL embedding path and fp32 arithmetic.
override WGX : u32 = 64u;
override WGY : u32 = 1u;
override WGZ : u32 = 1u;

@group(0) @binding(0) var<storage, read>       src : array<f32>;
@group(0) @binding(1) var<storage, read_write> dst : array<f32>;

@compute @workgroup_size(WGX, WGY, WGZ)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    if (i < arrayLength(&src)) {
        dst[i] = src[i] * 2.0 + 1.0;
    }
}
