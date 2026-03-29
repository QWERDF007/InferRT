#include <inferrt/core/version.h>

#include <iostream>

int main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;
    std::cout << "InferRT version: " << inferrt::core::GetVersionString() << std::endl;
    std::cout << "InferRT branch: " << inferrt::core::GetBranchString() << std::endl;
    std::cout << "InferRT commit hash: " << inferrt::core::GetCommitHashString() << std::endl;
    std::cout << "InferRT full version: " << inferrt::core::GetFullVersionString() << std::endl;
    return 0;
}