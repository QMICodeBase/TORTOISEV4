#ifndef _REGISTRATION_SETTINGS_H
#define _REGISTRATION_SETTINGS_H

#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <boost/lexical_cast.hpp>
#include <fstream>
#include <iostream>

class RegistrationSettings
{
public:
    static RegistrationSettings& get()
    {
        static RegistrationSettings instance;
        return instance;
    }

    void parseFile(std::string settings_file)
    {
        std::ifstream infile(settings_file.c_str());
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

                this->settings[var_name]=var_value;
            }
        }
        infile.close();

        std::ifstream input_file(settings_file.c_str());

        std::string all_file((std::istreambuf_iterator<char>(input_file)), std::istreambuf_iterator<char>());
        input_file.close();


        std::map<std::string, std::string>::iterator it;
        for (it = settings.begin(); it != settings.end(); it++)
        {
            std::string key=  it->first;
            int pos = all_file.find(std::string("<"+key));

            int pos2= all_file.rfind("</",pos);
            int pos3= all_file.find("\n",pos2);

            comments[key] = all_file.substr(pos3+1, pos-pos3 -1);
        }
        input_file.close();


    }

    template<typename _T>
    void setValue(std::string key,_T val)
    {

        std::string value_str= boost::lexical_cast<std::string>(val);
        settings[key]= value_str;
    }

    template<typename _T>
    void setVectorValue(std::string key,std::vector<_T> val)
    {
        if(val.size()>0)
        {
            std::string value_str=boost::lexical_cast<std::string>(val[0]);

            for(int s=1;s<val.size();s++)
                value_str=value_str + std::string(",") + boost::lexical_cast<std::string>(val[s]);
            settings[key]= value_str;
        }
        else
        {
            settings[key]= "";
        }
    }




    template<typename _T>
    _T getValue(std::string key)
    {
        std::string value_str= settings[key];

        _T val;
        try
        {
            val=boost::lexical_cast<_T>(value_str);
            return val;
        }
        catch(boost::bad_lexical_cast &)
        {
            val = _T();
            std::cout<<"Key " << key << " does not exist in the registration settings... Assuming value: " <<val <<std::endl;
            return val;
        }
    }


    template<typename _T>
    std::vector<_T> getVectorValue(std::string key)
    {
        std::string value_str= settings[key];        

        std::vector<_T> val;
        if(value_str=="")
            return val;

        try
        {
            int npos=value_str.find(",");
            while(npos!=std::string::npos)
            {
                std::string ss = value_str.substr(0,npos);
                _T cval=boost::lexical_cast<_T>(ss);
                val.push_back(cval);
                value_str= value_str.substr(npos+1);
                npos=value_str.find(",");
            }
            val.push_back(boost::lexical_cast<_T>(value_str));
            return val;
        }
        catch(boost::bad_lexical_cast &)
        {            
            std::cout<<"Key " << key << " does not exist in the registration settings... "  <<std::endl;
            return val;
        }
    }

    std::map<std::string,std::string> GetSettings(){return settings;};
    std::map<std::string,std::string> GetComments(){return comments;};


private:
    RegistrationSettings(){};
    RegistrationSettings(const RegistrationSettings&);
    RegistrationSettings& operator=(const RegistrationSettings&);

    std::map<std::string,std::string> settings;
    std::map<std::string,std::string> comments;
};
#endif


//std::ifstream input("somefile.txt");
//ConfigStore::get().parseFile(input);
//std::cout<<ConfigStore::get().getValue<std::string>(std::string("thing"))<<std::endl;
