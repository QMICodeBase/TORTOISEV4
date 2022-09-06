#include "gradcal.h"


void print_keys(std::vector<int> key){
    int nx = key.size();
    for (int i =0; i < nx; i++){
        std::cout << "Key" << i << ": " <<key.at(i) << std::endl;
    }
}

void print_coef(std::vector<float> key){
    int nx = key.size();
    for (int i =0; i < nx; i++){
        std::cout << "Key" << i << ": " <<key.at(i) << std::endl;
    }
}

int main(int argc, char * argv[]){
    std::string gradFn;
    // Check if user input correctly
    if (argc < 2){
        std::cout << "Usage: read_grad_cal gradFileName"  << std::endl;
        return EXIT_FAILURE;
    }

    gradFn = argv[1];
    GRADCAL *outputGrad = new GRADCAL(gradFn);
    std::vector <int> Xkey = outputGrad->get_X_key();
    print_keys(Xkey);
    std::vector <int> Ykey = outputGrad->get_Y_key();
    print_keys(Ykey);
    std::vector <int> Zkey = outputGrad->get_Z_key();
    print_keys(Zkey);

    std::vector <float> X_coef = outputGrad->get_X_coef();
    print_coef(X_coef);
    std::vector <float> Y_coef = outputGrad->get_Y_coef();
    print_coef(Y_coef);
    std::vector <float> Z_coef = outputGrad->get_Y_coef();
    print_coef(Z_coef);










    return 1;
}
