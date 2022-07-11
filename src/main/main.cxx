#include <iostream>
#include "TORTOISE.h"
#include <chrono>

int main(int argc, char * argv[])
{

    std::chrono::steady_clock::time_point Tbegin = std::chrono::steady_clock::now();
    TORTOISE(argc,argv);    

    std::chrono::steady_clock::time_point Tend = std::chrono::steady_clock::now();

    std::cout << "TOTAL runtime: " << std::chrono::duration_cast<std::chrono::minutes> (Tend - Tbegin).count() << "mins" << std::endl;

    return EXIT_SUCCESS;
}

