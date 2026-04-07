#include <inferrt/core/version.h>

#include <iostream>

int main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;
    std::cout << "InferRT version: " << irt::core::GetVersionString() << std::endl;
    std::cout << "InferRT branch: " << irt::core::GetBranchString() << std::endl;
    std::cout << "InferRT commit hash: " << irt::core::GetCommitHashString() << std::endl;
    std::cout << "InferRT full version: " << irt::core::GetFullVersionString() << std::endl;
    std::cout << "InferRT build time: " << irt::core::GetBuildTimeString() << std::endl;
    return 0;
}