//#ifndef _READ_GRAD_CAL_H
//#define _READ_GRAD_CAL_H

//#include "iostream"
//#include <vector>
//#include <string>
//#include "defines.h"
//#include "cmath"

////gradcal_coef
//struct gradCal{
//    std::string gradType;
//    float R0 = 250.;
//    std::vector<float> gradX_coef;
//    std::vector<float> gradY_coef;
//    std::vector<float> gradZ_coef;

//    std::vector<int>Xkeys;
//    std::vector<int>Ykeys;
//    std::vector<int>Zkeys;

// };

//float bytesToFLoat(char b0, char b1, char b2, char b3){
//    float output;
//    *((char*)(&output) + 3) = b3;
//    *((char*)(&output) + 2) = b2;
//    *((char*)(&output) + 1) = b1;
//    *((char*)(&output) + 0) = b0;
//    return output;
//}

//std::vector<float> byateArrtoFloatArr(char b[], int numFloat){
//    float temp;
//    std::vector<float> B;
//    for (int i=0; i < numFloat; i++){
//        temp = bytesToFLoat(b[i*numFloat + 0],b[i*numFloat + 1],b[i*numFloat + 2],b[i*numFloat + 3]);
//        B.push_back(temp);
//    }
//    return B;
//}

//std::vector<int> convertShortInttoInt(short int b[], int numInt){
//    int temp;
//    std::vector <int> B;
//    for (int i = 0; i < numInt; i++){
//        temp = int(b[i]);
//        B.push_back(temp);
//    }
//    return B;
//}



//class GRADCAL{
//public:
//    //Constructor
//    GRADCAL(){
//        std::string gradFilename = NULL;
//        float R0 = 250.;
//    }

//    GRADCAL(std::string fileName){
//        std::string gradFilename = fileName;
//        this->initFile(fileName);
//    }

//    // Destructor
//    ~GRADCAL();

//    //Init Input Gradcal
//    void initFile(std::string gradFilename){

//        // Assume R0 = 250;
//        float R0 = 250.;
//        // Check if file exist
//        if (!fs::exists(fs::path(gradFilename))){
//            std::cout << "File does not exist" << std::endl;
//        }

//        //Load Binary File
//        std::fstream gradFile (gradFilename, std::ios::in | std::ios::binary);
//        if (!gradFile){
//            std::cout << "Cannot open file!" << std::endl;

//        }


//        if (gradFilename.substr(gradFilename.find_last_of(".") + 1 )=="gc"){
//            std::cout <<"Input in.ASB type" << std::endl;
//            this->read_gradCal_asb(gradFile);


//        }
//        else if  (gradFilename.substr(gradFilename.find_last_of(".") + 1 )=="dat"){
//            std::cout <<"Input in .dat type" << std::endl;
//            this->read_gradCal_dat(gradFile);

//        }
//        else if  (gradFilename.substr(gradFilename.find_last_of(".") + 1 )=="grad"){
//            std::cout <<"Input in .grad type" << std::endl;
//            this->read_gradCal_grad(gradFile);

//        }
//        else {
//          std::cout << "Not grad file" << std::endl;
//        }
//    }
//    // Accessible funtions
//    std::vector<int>get_X_key(){
//        return this->grads_cal.Xkeys;
//    }
//    std::vector<int>get_Y_key(){
//        return this->grads_cal.Ykeys;
//    }
//    std::vector<int>get_Z_key(){
//        return this->grads_cal.Zkeys;
//    }
//    std::vector<float>get_X_coef(){
//        return this->grads_cal.gradX_coef;
//    }
//    std::vector<float>get_Y_coef(){
//        return this->grads_cal.gradY_coef;
//    }
//    std::vector<float>get_Z_coef(){
//        return this->grads_cal.gradZ_coef;
//    }


//private:
//    // Not accessible var.


//    gradCal grads_cal;
//    std::string gradFilename;
//    void read_gradCal_grad(std::fstream &gradFile, float R0 = 250){
//        std::string input;
//        // grad values are initialized
//        grads_cal.R0 = R0;
//        grads_cal.gradZ_coef.push_back(1.);
//        grads_cal.Xkeys.push_back(1);
//        grads_cal.Xkeys.push_back(1);
//        grads_cal.Ykeys.push_back(1);
//        grads_cal.Ykeys.push_back(-1);
//        grads_cal.Zkeys.push_back(1);
//        grads_cal.Zkeys.push_back(0);

//        size_t posA1, posA2, posA3;
//        int tempidx, idx1, idx2;
//        float coeftemp;
//        while (!gradFile.eof()){
//            std::getline(gradFile, input,'\n');
//            //position of A(int, int) or  B(int, int)
//            posA1 = input.find_first_of("(", 3, 3);
//            if ((posA1 != std::string::npos) & (posA1 <10))
//            {
//                // Check if before the "(" is integer. Else, continue.
//                try {
//                    tempidx = std::stoi(input.substr(0, posA1));
//                }
//                catch (const std::exception& e){
//                    throw e;
//                    continue;
//                }
//                posA2 = input.find(",");
//                posA3 = input.find(")");
//                if ((posA2 == std::string::npos) | (posA3 == std::string::npos)){
//                      continue;
//                  }

//                idx1 = std::stoi(input.substr(posA1+2, posA2 - posA1-2));
//                idx2 = std::stoi(input.substr(posA2+1, posA3 - posA2-1));

//                coeftemp = std::stof(input.substr(posA3+1, input.size() - posA3 -2));

//                if (input.find("x")!= std::string::npos){
//                    grads_cal.gradX_coef.push_back(coeftemp);
//                    grads_cal.Xkeys.push_back(idx1);
//                    grads_cal.Xkeys.push_back(idx2);
//                }
//                else if (input.find("y") != std::string::npos){
//                    idx2 = -1*idx2;
//                    grads_cal.gradY_coef.push_back(coeftemp);
//                    grads_cal.Ykeys.push_back(idx1);
//                    grads_cal.Ykeys.push_back(idx2);
//                }
//                else if(input.find("z") != std::string::npos){
//                    grads_cal.gradZ_coef.push_back(coeftemp);
//                    grads_cal.Zkeys.push_back(idx1);
//                    grads_cal.Zkeys.push_back(idx2);
//                }
//            }
//        }
//    }
//    void read_gradCal_dat(std::fstream &gradFile, float R0 = 250){
//        std::string input;

//        float rge = 10.;
//        std::vector<float> renorm(10,1.);
//        std::vector<float> xynorm(10,1.);
//        std::vector<float> znorm(10,1.);
//        float temp;
//        int tempind;

//        xynorm[1] = 1./sqrt(3.);
//        xynorm[2] = sqrt(8./3.);
//        xynorm[3] = sqrt(8./5);
//        xynorm[4] = sqrt(64./15);

//        znorm[1] = 2;
//        znorm[2] = 2;
//        znorm[3] = 8;
//        znorm[4] = 8;

//        //create xynorm, znorm.
//        for (int i=0; i <10 ; i++){
//            temp = pow(R0/rge,i);
//            xynorm[i] = xynorm[i] *temp;
//            znorm[i] = znorm[i] *temp;
//            renorm[i] = temp;
//        }
//        while (!gradFile.eof()){
//            std::getline(gradFile, input,'\n');

//            if (input.compare(0,6, "SCALEX") == 0){

//                temp = std::stof(input.substr(10));
//                if ((temp == 0)){
//                    if ((grads_cal.gradX_coef.size() ==0)){
//                        temp += 1;
//                    }
//                    else continue;
//                }
//                tempind = std::stoi(input.substr(6,7));
//                grads_cal.gradX_coef.push_back(temp*xynorm[tempind]);
//                grads_cal.Xkeys.push_back(tempind);
//                grads_cal.Xkeys.push_back(1);
//            }

//            if (input.compare(0,6, "SCALEY") == 0){

//                temp = std::stof(input.substr(10));
//                if ((temp == 0)){
//                    if ((grads_cal.gradY_coef.size() ==0)){
//                        temp += 1;
//                    }
//                    else continue;
//                }
//                tempind = std::stoi(input.substr(6,7));
//                grads_cal.gradY_coef.push_back(temp * xynorm[tempind]);
//                grads_cal.Ykeys.push_back(tempind);
//                grads_cal.Ykeys.push_back(1);
//            }

//            if (input.compare(0,6, "SCALEZ") == 0){

//                temp = std::stof(input.substr(10));
//                if ((temp == 0)){
//                    if ((grads_cal.gradZ_coef.size() ==0)){
//                        temp += 1;
//                    }
//                    else continue;
//                }
//                tempind = std::stoi(input.substr(6,7));
//                grads_cal.gradZ_coef.push_back(temp * znorm[tempind]);
//                grads_cal.Zkeys.push_back(tempind);
//                grads_cal.Zkeys.push_back(1);
//            }
//        }
//    }
//    void read_gradCal_asb(std::fstream &gradFile, float R0 = 250.){
//        // change gradFile to string name.


//        short int gradSize[3];
//        gradFile.seekg(0, std::ios::beg);
//        gradFile.read(reinterpret_cast<char *> (&gradSize[0]), 6);
//        int nx = gradSize[0];
//        int ny = gradSize[1];
//        int nz = gradSize[2];

//        char byte_xcoef[4*nx], byte_ycoef[4*ny],byte_zcoef[4*nz];
//        short int xkey[nx*2], ykey[ny*2], zkey[nz*2];
//        float r0;
//        float shim[3];

//        gradFile.read(reinterpret_cast<char *>(&byte_xcoef[0]), sizeof(byte_xcoef));
//        gradFile.read(reinterpret_cast<char *>(&xkey[0]),sizeof(xkey));
//        grads_cal.Xkeys = convertShortInttoInt(xkey, 2*nx);
//        grads_cal.gradX_coef = byateArrtoFloatArr(byte_xcoef,nx);

//        gradFile.read(reinterpret_cast<char *>(&byte_ycoef[0]), sizeof(byte_ycoef));
//        gradFile.read(reinterpret_cast<char *>(&ykey[0]), sizeof(ykey));
//        grads_cal.Ykeys = convertShortInttoInt(ykey, 2*nx);
//        grads_cal.gradY_coef = byateArrtoFloatArr(byte_ycoef,ny);


//        gradFile.read(reinterpret_cast<char *>(&byte_zcoef[0]), sizeof(byte_zcoef));
//        gradFile.read(reinterpret_cast<char *>(&zkey[0]), sizeof(zkey));
//        grads_cal.Zkeys = convertShortInttoInt(zkey, 2*nx);
//        grads_cal.gradZ_coef = byateArrtoFloatArr(byte_zcoef,nz);


////         Convert R0 Bytes to float
//        gradFile.read(reinterpret_cast<char *>(&byte_xcoef[0]), sizeof(byte_xcoef));
//        r0 = bytesToFLoat(byte_xcoef[0], byte_xcoef[1], byte_xcoef[2], byte_xcoef[3] );
//        grads_cal.R0 = r0;
//        grads_cal.gradType = "asb";



//}

//};

//#endif
