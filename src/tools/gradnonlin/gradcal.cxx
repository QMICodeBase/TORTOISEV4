#include "gradcal.h"
#include "defines.h"

float bytesToFLoat(char b0, char b1, char b2, char b3){
    float output;
    *((char*)(&output) + 3) = b3;
    *((char*)(&output) + 2) = b2;
    *((char*)(&output) + 1) = b1;
    *((char*)(&output) + 0) = b0;
    return output;
}

std::vector<double> byateArrtoFloatArr(char b[], int numFloat){
    float temp;
    std::vector<double> B;
    for (int i=0; i < numFloat; i++){
        temp = bytesToFLoat(b[i*numFloat + 0],b[i*numFloat + 1],b[i*numFloat + 2],b[i*numFloat + 3]);
        B.push_back((temp));
    }
    return B;
}

std::vector<int> convertShortInttoInt(short int b[], int numInt){
    int temp;
    std::vector <int> B;
    for (int i = 0; i < numInt; i++){
        temp = int(b[i]);
        B.push_back(temp);
    }
    return B;
}


GRADCAL::GRADCAL(){
    GradCoef grads_cal;
    std::string gradFilename = "";
    float R0 = 250.;
    GRADCAL::init();
}

void GRADCAL::init(){
    //this->grads_cal.Xkeys = {1,1,3,1,5,1,7,1};
    //this->grads_cal.Ykeys = {1,-1,3,-1, 5,-1,7,-1};
    //this->grads_cal.Zkeys = {1,0,3,0,5,0,7,0};

    this->grads_cal.Xkeys = {1,1,3,1,5,1,7,1,9,1};
    this->grads_cal.Ykeys = {1,-1,3,-1, 5,-1,7,-1,9,-1};
    this->grads_cal.Zkeys = {1,0,3,0,5,0,7,0,9,0};
    this->grads_cal.gradX_coef = std::vector<double> (this->grads_cal.Xkeys.size()/2, 0);
    this->grads_cal.gradY_coef = std::vector<double> (this->grads_cal.Ykeys.size()/2, 0);
    this->grads_cal.gradZ_coef = std::vector<double> (this->grads_cal.Zkeys.size()/2, 0);
    this->grads_cal.R0 = 250.0;

}

GRADCAL::GRADCAL(std::string fileName){
    std::string gradFilename = fileName;
    this->read_grad_file(fileName);

}

GRADCAL::~GRADCAL(){


}

void GRADCAL::read_grad_file(std::string gradFilename){
    // Assume R0 = 250;
    ;
    // Check if file exist
    if (!fs::exists(fs::path(gradFilename))){
        std::cout << "File does not exist" << std::endl;
    }


    if (gradFilename.substr(gradFilename.find_last_of(".") + 1 )=="gc"){
        std::cout <<"Gradient nonlinearity file in TORTOISE format" << std::endl;
        this->read_ASB_format(gradFilename);



    }
    else if  (gradFilename.substr(gradFilename.find_last_of(".") + 1 )=="dat"){
        std::cout <<"Gradient nonlinearity file in GE format" << std::endl;
        this->read_GE_format(gradFilename);

    }
    else if  (gradFilename.substr(gradFilename.find_last_of(".") + 1 )=="grad"){
        std::cout <<"Gradient nonlinearity file in SIEMENS format" << std::endl;
        this->read_Siemens_format(gradFilename);

    }
    else {
      std::cout << "Not grad file" << std::endl;
    }

}

void GRADCAL::read_Siemens_format(std::string gradFilename){
    //read file .grad
    std::fstream gradFile (gradFilename, std::ios::in | std::ios::binary);
    if (!gradFile){
        std::cout << "Cannot open file!" << std::endl;}
    std::string input;
//    grads_cal.R0 = R0;
    grads_cal.gradZ_coef.push_back(1.);
    grads_cal.gradY_coef.push_back(1.);
    grads_cal.gradX_coef.push_back(1.);

    grads_cal.Xkeys.push_back(1);
    grads_cal.Xkeys.push_back(1);
    grads_cal.Ykeys.push_back(1);
    grads_cal.Ykeys.push_back(-1);
    grads_cal.Zkeys.push_back(1);
    grads_cal.Zkeys.push_back(0);


    size_t posA1, posA2, posA3;
    int tempidx, idx1, idx2;
    float coeftemp;
    float R0=250;
    while (!gradFile.eof())
    {
        std::getline(gradFile, input,'\n');

        int okpos= input.find("= R0");

        if(okpos!=std::string::npos)
        {
            std::string sub= input.substr(1,5);
            R0=atof(sub.c_str())*1000.;
        }

        //position of A(int, int) or  B(int, int)                        
        posA1 = input.find_first_of("(", 3, 3);
        if ((posA1 != std::string::npos) && (posA1 <10))
        {
            // Check if before the "(" is integer. Else, continue.
            /*
            try {
                tempidx = std::stoi(input.substr(0, posA1));
            }
            catch (const std::exception& e){
                throw e;
                continue;
            }
            */
            posA2 = input.find(",");
            posA3 = input.find(")");
            if ((posA2 == std::string::npos) | (posA3 == std::string::npos)){
                  continue;
              }

            std::string m1=input.substr(posA1+1, posA2 - posA1-1);
            std::string m2=input.substr(posA2+1, posA3 - posA2-1);

            idx1 = std::stoi(m1);
            idx2 = std::stoi(m2);

            coeftemp = std::stof(input.substr(posA3+1, input.size() - posA3 -2));

            if (input.find("x")!= std::string::npos){
                grads_cal.gradX_coef.push_back(coeftemp);
                grads_cal.Xkeys.push_back(idx1);
                grads_cal.Xkeys.push_back(idx2);
            }
            else if (input.find("y") != std::string::npos){
                idx2 = -1*idx2;
                grads_cal.gradY_coef.push_back(coeftemp);
                grads_cal.Ykeys.push_back(idx1);
                grads_cal.Ykeys.push_back(idx2);
            }
            else if(input.find("z") != std::string::npos){
                grads_cal.gradZ_coef.push_back(coeftemp);
                grads_cal.Zkeys.push_back(idx1);
                grads_cal.Zkeys.push_back(idx2);
            }
        }
    }
    grads_cal.R0=R0;
    gradFile.close();
    grads_cal.gradType="siemens";

}
void GRADCAL::read_GE_format(std::string gradFilename){

    // read file .dat
    std::fstream gradFile (gradFilename, std::ios::in | std::ios::binary);
    if (!gradFile){
        std::cout << "Cannot open file!" << std::endl;}
    std::string input;

    float rge = 10.;
    std::vector<double> renorm(10,1.);
    std::vector<double> xynorm(10,1.);
    std::vector<double> znorm(10,1.);
    float temp;
    int tempind;

    xynorm[1] = 1./sqrt(3.);
    xynorm[2] = sqrt(8./3.);
    xynorm[3] = sqrt(8./5);
    xynorm[4] = sqrt(64./15);

    znorm[1] = 2;
    znorm[2] = 2;
    znorm[3] = 8;
    znorm[4] = 8;
    float R0 = grads_cal.R0;
    //create xynorm, znorm.
    for (int i=0; i <10 ; i++){
        temp = pow(R0/rge,i);
        xynorm[i] = xynorm[i] *temp;
        znorm[i] = znorm[i] *temp;
        renorm[i] = temp;
    }
    while (!gradFile.eof()){
        std::getline(gradFile, input,'\n');

        if (input.compare(0,6, "SCALEX") == 0){

            temp = std::stof(input.substr(10));
            if ((temp == 0)){
                if ((grads_cal.gradX_coef.size() ==0)){
                    temp += 1;
                }
                else continue;
            }
            tempind = std::stoi(input.substr(6,7));
            grads_cal.gradX_coef.push_back(temp*xynorm[tempind-1]);
            grads_cal.Xkeys.push_back(tempind);
            grads_cal.Xkeys.push_back(1);
        }

        if (input.compare(0,6, "SCALEY") == 0){

            temp = std::stof(input.substr(10));
            if ((temp == 0)){
                if ((grads_cal.gradY_coef.size() ==0)){
                    temp += 1;
                }
                else continue;
            }
            tempind = std::stoi(input.substr(6,7));
            grads_cal.gradY_coef.push_back(temp * xynorm[tempind-1]);
            grads_cal.Ykeys.push_back(tempind);
            grads_cal.Ykeys.push_back(-1);
        }

        if (input.compare(0,6, "SCALEZ") == 0){

            temp = std::stof(input.substr(10));
            if ((temp == 0)){
                if ((grads_cal.gradZ_coef.size() ==0)){
                    temp += 1;
                }
                else continue;
            }
            tempind = std::stoi(input.substr(6,7));
            grads_cal.gradZ_coef.push_back(temp * znorm[tempind-1]);
            grads_cal.Zkeys.push_back(tempind);
            grads_cal.Zkeys.push_back(0);
        }
    }
    grads_cal.gradType="ge";
}

void GRADCAL::read_ASB_format(std::string gradFilename)
{

    FILE *fp = fopen(gradFilename.c_str(), "rb");
    if(!fp)
    {
        std::cout << "Cannot open file!" << gradFilename<<std::endl;
    }

    short nx,ny,nz;
    fread(&nx,sizeof(short),1,fp);
    fread(&ny,sizeof(short),1,fp);
    fread(&nz,sizeof(short),1,fp);

    short int xkey[nx*2], ykey[ny*2], zkey[nz*2];
    float r0;
    float xcoef[nx], ycoef[ny], zcoef[nz];

    fread(xcoef,sizeof(float),nx,fp);
    fread(xkey,sizeof(short),2*nx,fp);
    fread(ycoef,sizeof(float),ny,fp);
    fread(ykey,sizeof(short),2*ny,fp);
    fread(zcoef,sizeof(float),nz,fp);
    fread(zkey,sizeof(short),2*nz,fp);

    fread(&r0,sizeof(float),1,fp);
    fclose(fp);


    {
        std::vector<double> coeffs;
        std::vector<int> keys;
        for(int i=0;i<nx;i++)
        {
            coeffs.push_back(xcoef[i]);
            keys.push_back(xkey[2*i]);
            keys.push_back(xkey[2*i+1]);
        }
        grads_cal.Xkeys = keys;
        grads_cal.gradX_coef =coeffs;
    }
    {
        std::vector<double> coeffs;
        std::vector<int> keys;
        for(int i=0;i<ny;i++)
        {
            coeffs.push_back(ycoef[i]);
            keys.push_back(ykey[2*i]);
            keys.push_back(ykey[2*i+1]);
        }
        grads_cal.Ykeys = keys;
        grads_cal.gradY_coef =coeffs;
    }
    {
        std::vector<double> coeffs;
        std::vector<int> keys;
        for(int i=0;i<nz;i++)
        {
            coeffs.push_back(zcoef[i]);
            keys.push_back(zkey[2*i]);
            keys.push_back(zkey[2*i+1]);
        }
        grads_cal.Zkeys = keys;
        grads_cal.gradZ_coef =coeffs;
    }
    grads_cal.R0 = r0;
    grads_cal.gradType = "asb";




    /*

    // read file .gc
    std::fstream gradFile (gradFilename, std::ios::in | std::ios::binary);
    if (!gradFile){
        std::cout << "Cannot open file!" << std::endl;}
    short int gradSize[3];
    gradFile.seekg(0, std::ios::beg);
    gradFile.read(reinterpret_cast<char *> (&gradSize[0]), 6);
    int nx = gradSize[0];
    int ny = gradSize[1];
    int nz = gradSize[2];

    char byte_xcoef[4*nx], byte_ycoef[4*ny],byte_zcoef[4*nz];
    short int xkey[nx*2], ykey[ny*2], zkey[nz*2];
    float r0;

    float xcoef[nx], ycoef[ny], zcoef[nz];

    std::cout<<sizeof(byte_xcoef)<<std::endl;
    gradFile>> xcoef;
    gradFile>> xkey;
    gradFile>> ycoef;
    gradFile>> ykey;
    gradFile>> zcoef;
    gradFile>> zkey;
    gradFile>> r0;

    gradFile.close();



    gradFile.read(reinterpret_cast<char *>(&byte_xcoef[0]), sizeof(byte_xcoef));
    gradFile.read(reinterpret_cast<char *>(&xkey[0]),sizeof(xkey));
    grads_cal.Xkeys = convertShortInttoInt(xkey, 2*nx);
    grads_cal.gradX_coef = byateArrtoFloatArr(byte_xcoef,nx);

    gradFile.read(reinterpret_cast<char *>(&byte_ycoef[0]), sizeof(byte_ycoef));
    gradFile.read(reinterpret_cast<char *>(&ykey[0]), sizeof(ykey));
    grads_cal.Ykeys = convertShortInttoInt(ykey, 2*nx);
    grads_cal.gradY_coef = byateArrtoFloatArr(byte_ycoef,ny);


    gradFile.read(reinterpret_cast<char *>(&byte_zcoef[0]), sizeof(byte_zcoef));
    gradFile.read(reinterpret_cast<char *>(&zkey[0]), sizeof(zkey));
    grads_cal.Zkeys = convertShortInttoInt(zkey, 2*nx);
    grads_cal.gradZ_coef = byateArrtoFloatArr(byte_zcoef,nz);

    gradFile.read(reinterpret_cast<char *>(&byte_xcoef[0]), sizeof(byte_xcoef));
    r0 = bytesToFLoat(byte_xcoef[0], byte_xcoef[1], byte_xcoef[2], byte_xcoef[3] );
    grads_cal.R0 = r0;
    grads_cal.gradType = "asb";
    */

}

void GRADCAL::write_ASB_format(GradCoef gradscal, std::string outfn){
    std::fstream file;
    file.open(outfn, std::ios::out | std::ios::binary);
    int nx = gradscal.Xkeys.size()/2;
    int ny = gradscal.Ykeys.size()/2;
    int nz = gradscal.Zkeys.size()/2;
    file.write(reinterpret_cast<char *>(&nx), 2);
    file.write(reinterpret_cast<char *>(&ny), 2);
    file.write(reinterpret_cast<char *>(&nz), 2);

    for (int i  = 0; i < nx ;i++){
        file.write(reinterpret_cast<char *>(&gradscal.gradX_coef[i]), 4);
    }
    for (int i  = 0; i < nx ;i++){
        file.write(reinterpret_cast<char *>(&gradscal.Xkeys[2*i]), 2);
        file.write(reinterpret_cast<char *>(&gradscal.Xkeys[2*i+1]), 2);
    }
    for (int i  = 0; i < ny ;i++){
        file.write(reinterpret_cast<char *>(&gradscal.gradY_coef[i]), 4);
    }
    for (int i  = 0; i < ny ;i++){
        file.write(reinterpret_cast<char *>(&gradscal.Ykeys[2*i]), 2);
        file.write(reinterpret_cast<char *>(&gradscal.Ykeys[2*i+1]), 2);

    } for (int i  = 0; i < nz ;i++){
        file.write(reinterpret_cast<char *>(&gradscal.gradZ_coef[i]), 4);
    }
    for (int i  = 0; i < nz ;i++){
        file.write(reinterpret_cast<char *>(&gradscal.Zkeys[2*i]), 2);
        file.write(reinterpret_cast<char *>(&gradscal.Zkeys[2*i+1]), 2);
    }
    file.write(reinterpret_cast<char *>(&gradscal.R0),4);
    file.close();
}



