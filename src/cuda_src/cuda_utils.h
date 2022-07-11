#ifndef _CUDA_UTILS_H
#define _CUDA_UTILS_H


#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>



/********************/
/* CUDA ERROR CHECK */
/********************/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code,  std::string file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file.c_str(), line);
        if (abort)
        {
            getchar();
            exit(code);
        }
    }
}





inline void copy3DPitchedPtrToHost(cudaPitchedPtr _src, float *_dst, int width, int height, int depth)

{

  cudaExtent copy_extent = make_cudaExtent(width*sizeof(float),height,depth);

  cudaMemcpy3DParms copyParams = {0};

  float *h_dest = _dst;

  copyParams.srcPtr = _src;

  copyParams.dstPtr = make_cudaPitchedPtr((void*)h_dest, width*sizeof(float), width, height);

  copyParams.kind = cudaMemcpyDeviceToHost;

  copyParams.extent = copy_extent;

  gpuErrchk(cudaMemcpy3D(&copyParams));



}

inline void copy3DHostToPitchedPtr(float *_src, cudaPitchedPtr _dst, int width, int height, int depth)

{

  cudaExtent copy_extent = make_cudaExtent(width*sizeof(float),height,depth);

  cudaMemcpy3DParms copyParams = {0};

  float *h_source = _src;

  copyParams.srcPtr = make_cudaPitchedPtr((void*)h_source, copy_extent.width, copy_extent.width/sizeof(float), copy_extent.height);

  copyParams.dstPtr = _dst;

  copyParams.kind = cudaMemcpyHostToDevice;

  copyParams.extent = copy_extent;

  gpuErrchk(cudaMemcpy3D(&copyParams));


}






#endif
