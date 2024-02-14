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
#ifdef USECUDA
        for(int i=0;i<8;i++)
            gpu_available[i]=1;
#endif
    }



#ifdef USECUDA
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
    }
    static void ReleaseITKThreadFor()
    {
        EnableOMPThread();
    }


private:
    uint NMaxCores;
    static std::atomic_uint NAvailableCores;
    static std::vector<uint> Nthreads_per_OMP_thread;

#ifdef USECUDA
    static std::array< std::atomic_bool,8 > gpu_available;
#endif



};



#endif
