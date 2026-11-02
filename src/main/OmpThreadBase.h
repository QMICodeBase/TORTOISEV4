#ifndef _OMPTHREADBASE_H
#define _OMPTHREADBASE_H


#include <omp.h>
#include <chrono>
#include <thread>
#include <atomic>
#include "../utilities/TORTOISE_Utilities.h"



class OMPTHREADBASE
{
private:

 
public:
    void SetNMaxCores(int nc){NMaxCores=nc;}
    void static SetNAvailableCores(int nc){NAvailableCores=nc;}
    int  GetNMaxCores(){return NMaxCores;}
    static int  GetNAvailableCores(){return NAvailableCores;}

    void static SetThreadArray(std::vector<uint>thread_array)
    {
        Nthreads_per_OMP_thread=thread_array;
    }
    OMPTHREADBASE()
    {
#ifdef USEGPU
        for(int i=0;i<8;i++)
            gpu_available[i]=1;
#endif
    }



#ifdef USEGPU
    void static ReserveGPU(int id)
    {                                
        if(id==0)
        {
            #pragma omp critical
            {
                while(!gpu_available[id])
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));

                gpu_available[id]=false;
            }
        }
        if(id==1)
        {
            #pragma omp critical
            {
                while(!gpu_available[id])
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));

                gpu_available[id]=false;
            }
        }
        if(id==2)
        {
            #pragma omp critical
            {
                while(!gpu_available[id])
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));

                gpu_available[id]=false;
            }
        }
        if(id==3)
        {
            #pragma omp critical
            {
                while(!gpu_available[id])
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));

                gpu_available[id]=false;
            }
        }



    }
    void static ReleaseGPU(int id)
    {
    //    #pragma omp critical
        {
            gpu_available[id]=true;
        }
    }
#endif



    void static EnableOMPThread()
    {
        #pragma omp critical
        {
            int id =omp_get_thread_num();
            Nthreads_per_OMP_thread[id]=1;
        }
    }
    void static DisableOMPThread()
    {
        #pragma omp critical
        {
            int id =omp_get_thread_num();
            Nthreads_per_OMP_thread[id]=0;
        }
    }

    static int GetAvailableITKThreadFor()
    {
#ifdef TORTOISE_DETERMINISTIC_GPU
        // VALIDATION BUILD ONLY. The value below is normally derived from how many
        // OMP threads happen to be active at this instant, so the SAME volume can
        // be given a different ITK work-unit count on different runs. That count
        // decides how ITK splits the image for its metric's per-thread partial
        // sums, and float addition is not associative - so the metric value, and
        // hence the registration result, varies run to run.
        //
        // There is no RNG involved and nothing to seed: registration uses 100%
        // dense sampling (SetMetricSamplingPercentage(1.), no sampling strategy),
        // so this reduction-order effect is the whole mechanism.
        //
        // Routing DIFFPREP's volumes to the GPU (see DIFFPREP.cxx) removes this for
        // motion/eddy registration, but DRBUDDI's structural alignment
        // (DRBUDDI.cxx:1129-1130, MultiStartRigidSearch + RigidRegisterImagesEuler)
        // is ITK CPU code and was still varying - it was the first divergent
        // artefact (structural_used.nii) once DIFFPREP was made deterministic.
        //
        // A fixed count keeps full parallelism; it is only less adaptive to what
        // other threads are doing.
        if(NAvailableCores==0)
            NAvailableCores=getNCores();
        return (int)NAvailableCores;
#else
        int ma=0;

       // std::this_thread::sleep_for(std::chrono::milliseconds(5*id));
        #pragma omp critical
        {            
            if(NAvailableCores==0)
                NAvailableCores=getNCores();

            if(Nthreads_per_OMP_thread.size()==0)
            {
                ma=NAvailableCores;
            }
            else
            {

                int id =omp_get_thread_num();
                int total_threads=0;
                for(int t=0;t<Nthreads_per_OMP_thread.size();t++)
                    total_threads+=Nthreads_per_OMP_thread[t];

                if(total_threads<NAvailableCores)
                {
                   if(Nthreads_per_OMP_thread[id]==0 || total_threads==1)
                       ma=NAvailableCores;
                   else
                   {
                       Nthreads_per_OMP_thread[id]++;
                       ma=Nthreads_per_OMP_thread[id];
                   }
                }
                else
                    ma=1;
            }
        }

        return ma;
#endif
    }
    static void ReleaseITKThreadFor()
    {
        EnableOMPThread();
    }


private:
    uint NMaxCores;
    static std::atomic_uint NAvailableCores;
    static std::vector<uint> Nthreads_per_OMP_thread;

#ifdef USEGPU
    static std::array< std::atomic_bool,8 > gpu_available;
#endif



};



#endif
