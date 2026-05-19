#include <cxxopts.hpp>
#include <inferrt/core/version.h>

#include <iostream>

/**
 * @brief 打印 InferRT 的版本、分支、提交和构建时间信息。
 * @param argc 命令行参数个数。
 * @param argv 命令行参数数组。
 * @return 成功返回 0，失败返回非 0。
 */
int main(int argc, char *argv[])
{
    try
    {
        cxxopts::Options options(argv[0], "Print InferRT version information");
        options.add_options()("h,help", "Show help");

        const auto result = options.parse(argc, argv);
        if (result.count("help"))
        {
            std::cout << options.help() << std::endl;
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    std::cout << "InferRT version: " << irt::GetVersionString() << std::endl;
    std::cout << "InferRT branch: " << irt::GetBranchString() << std::endl;
    std::cout << "InferRT commit hash: " << irt::GetCommitHashString() << std::endl;
    std::cout << "InferRT full version: " << irt::GetFullVersionString() << std::endl;
    std::cout << "InferRT build time: " << irt::GetBuildTimeString() << std::endl;
    return 0;
}
