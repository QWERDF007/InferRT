#include <inferrt/core/version.h>

#include <iostream>

int main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;
    std::cout << "InferRT version: " << irt::GetVersionString() << std::endl;
    std::cout << "InferRT branch: " << irt::GetBranchString() << std::endl;
    std::cout << "InferRT commit hash: " << irt::GetCommitHashString() << std::endl;
    std::cout << "InferRT full version: " << irt::GetFullVersionString() << std::endl;
    std::cout << "InferRT build time: " << irt::GetBuildTimeString() << std::endl;
    return 0;
}