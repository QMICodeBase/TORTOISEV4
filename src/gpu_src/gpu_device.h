#ifndef _GPUSHIM_GPU_DEVICE_H
#define _GPUSHIM_GPU_DEVICE_H
// Backend-neutral device enumeration for DIFFPREP's GPU/CPU work split.
#include <string>
#include <vector>

#ifdef USEMETAL
    #include "../metal_src/metal_context.h"
    // As with WebGPU below: one device, selected at startup, so this is a no-op.
    inline std::vector<int> GPUDeviceIds() { mtlctx::Init(); return std::vector<int>(1, 0); }
    inline void GPUSetDevice(int) {}
#elif defined(USEWEBGPU)
    #include "../webgpu_src/webgpu_context.h"
    // The WebGPU backend binds exactly one adapter, chosen and verified at
    // startup (CLAUDE.md 3), so there is one device and selecting it is a no-op.
    inline std::vector<int> GPUDeviceIds() { wgpuctx::Init(); return std::vector<int>(1, 0); }
    inline void GPUSetDevice(int) {}
#else
    #include <cuda_runtime.h>
    inline std::vector<int> GPUDeviceIds()
    {
        int n = 0;
        cudaGetDeviceCount(&n);
        std::vector<int> ids;
        cudaDeviceProp prop;
        for(int g = 0; g < n; g++)
        {
            cudaGetDeviceProperties(&prop, g);
            std::string nm = prop.name;
            if(nm.find("Display") == std::string::npos)   // skip DGX display adaptors
                ids.push_back(g);
        }
        return ids;
    }
    inline void GPUSetDevice(int d) { cudaSetDevice(d); }
#endif
#endif
