#ifndef _MECC_SETTINGS_H
#define _MECC_SETTINGS_H

#include <string>
#include <fstream>
#include <sstream>
#include <boost/lexical_cast.hpp>


class MeccSettings
{

private:
    void parseFile(std::string settings_file)
    {
        std::ifstream infile(settings_file.c_str());
        if(!infile.is_open())
        {
            std::cout<<"Eddy current optimization settings file "<< settings_file << " does not exist... Exiting..."<<std::endl;
            exit(0);
        }


        std::string line;

        while (std::getline(infile, line))
        {
            line.erase(std::remove_if(line.begin(), line.end(), ::isspace),line.end());
            if(line != ""  && line.find("<!--")==std::string::npos)
            {
                int temp_pos1= line.find('>',1);
                int temp_pos2= line.find('<',1);

                std::string var_name = line.substr(1,temp_pos1-1);
                std::string var_value= line.substr(temp_pos1+1,temp_pos2-temp_pos1-1);

                if(var_name=="percent_of_max_available_cpus")
                {
                    pN_cpus=boost::lexical_cast<int>(var_value)/100.;
                    continue;
                }
                if(var_name=="DWIoptimizer")
                {
                    DWIoptimizer=var_value;
                    continue;
                }
                if(var_name=="T2optimizer")
                {
                    T2optimizer=var_value;
                    continue;
                }
                if(var_name=="nbins")
                {
                    nbins=boost::lexical_cast<int>(var_value);
                    continue;
                }
                if(var_name=="epsilon")
                {
                    epsilon=boost::lexical_cast<double>(var_value);
                    continue;
                }
                if(var_name=="brk_eps")
                {
                    brk_eps=boost::lexical_cast<double>(var_value);
                    continue;
                }
                if(var_name=="optimization_flags")
                {

                    bool val;
                    std::stringstream ss(var_value);
                    while (ss >> val)
                    {
                        optimization_flags.push_back(val);
                        if (ss.peek() == ',')
                            ss.ignore();
                    }
                    continue;
                }
                if(var_name=="init_grd_step")
                {

                    double val;
                    std::stringstream ss(var_value);
                    while (ss >> val)
                    {
                        init_grd_step.push_back(val);
                        if (ss.peek() == ',')
                            ss.ignore();
                    }
                    continue;
                }
                if(var_name=="num_grd_halve")
                {
                    n_grad_halve=boost::lexical_cast<int>(var_value);
                    continue;
                }
                if(var_name=="num_res")
                {
                    nres=boost::lexical_cast<int>(var_value);
                    continue;
                }
            }
        }
        infile.close();
    }


    int getNCores() {
    #ifdef WIN32
        SYSTEM_INFO sysinfo;
        GetSystemInfo(&sysinfo);
        return sysinfo.dwNumberOfProcessors;
    #elif MACOS
        int nm[2];
        size_t len = 4;
        uint32_t count;

        nm[0] = CTL_HW; nm[1] = HW_AVAILCPU;
        sysctl(nm, 2, &count, &len, NULL, 0);

        if(count < 1) {
            nm[1] = HW_NCPU;
            sysctl(nm, 2, &count, &len, NULL, 0);
            if(count < 1) { count = 1; }
        }
        return count;
    #else
        return sysconf(_SC_NPROCESSORS_ONLN);
    #endif
    }

public:
    MeccSettings()
    {         
         pN_cpus=0.75;
         DWIoptimizer= std::string("DIFFPREPOptimizer");
         T2optimizer= std::string("DIFFPREPOptimizer");
         nbins=100;
         epsilon=0.0001;
         brk_eps=0.0005;

         optimization_flags.resize(24);
         for(int i=0;i<6;i++)
             optimization_flags[i]=1;
         for(int i=6;i<24;i++)
             optimization_flags[i]=0;


/*
         init_grd_step.resize(21);
         init_grd_step[0]=2.5;
         init_grd_step[1]=2.5;
         init_grd_step[2]=2.5;
         init_grd_step[3]=0.04;
         init_grd_step[4]=0.04;
         init_grd_step[5]=0.04;
         init_grd_step[6]=0.02;
         init_grd_step[7]=0.02;
         init_grd_step[8]=0.02;
         init_grd_step[9]=0.002;
         init_grd_step[10]=0.002;
         init_grd_step[11]=0.002;
         init_grd_step[12]=0.0007;
         init_grd_step[13]=0.0007;
         init_grd_step[14]=0.0001;
         init_grd_step[15]=0.00001;
         init_grd_step[16]=0.00001;
         init_grd_step[17]=0.00001;
         init_grd_step[18]=0.00001;
         init_grd_step[19]=0.00001;
         init_grd_step[20]=0.00001;
*/
         n_grad_halve=5;
         nres=3;
    }

    MeccSettings(std::string fname){this->parseFile(fname);};


    MeccSettings(const MeccSettings&);
    MeccSettings& operator=(const MeccSettings&);

    int getNCpus()
    {
        if(pN_cpus==1)
            return 0;

         float nc =this->getNCores() *pN_cpus;
         return (int)round(nc);
    }

    inline std::vector<bool> getFlags(){return optimization_flags;};
    inline void setFlags(std::vector<bool> f){optimization_flags=f;};
    inline std::vector<double> getGrdSteps(){return init_grd_step;};
    inline int getNumberHalves(){return n_grad_halve;};
    inline double getBrkEps(){return brk_eps;};
    inline int getNBins(){return nbins;};
    inline std::string getDWIOptimizer() {return DWIoptimizer;};
    inline std::string getT2Optimizer() {return T2optimizer;};





private:
    float pN_cpus;
    std::string DWIoptimizer;
    std::string T2optimizer;
    int nbins;
    double epsilon;
    double brk_eps;

    std::vector<bool> optimization_flags;
    std::vector<double> init_grd_step;
    int n_grad_halve;
    int nres;
};


#endif

