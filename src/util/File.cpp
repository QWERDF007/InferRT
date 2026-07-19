#include <inferrt/util/File.hpp>
#include <inferrt/util/String.hpp>

#include <inferrt/core/Exception.hpp>

#include <fstream>
#include <utility>

namespace irt::util {

std::vector<std::string> readNonEmptyLines(const std::filesystem::path &file_path, bool trim_lines)
{
    std::ifstream input(file_path);
    if (!input)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to open text file: %s",
                             file_path.string().c_str());
    }

    std::vector<std::string> lines;
    std::string              line;
    while (std::getline(input, line))
    {
        if (trim_lines)
        {
            line = trim(std::move(line));
        }
        if (!line.empty())
        {
            lines.push_back(std::move(line));
        }
    }
    return lines;
}

void writeBinaryFile(const std::filesystem::path &file_path, const void *data, size_t num_bytes)
{
    std::ofstream output(file_path, std::ios::binary);
    if (!output)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to open output file: %s",
                             file_path.string().c_str());
    }
    output.write(static_cast<const char *>(data), static_cast<std::streamsize>(num_bytes));
    if (!output)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to write output file: %s",
                             file_path.string().c_str());
    }
}

} // namespace irt::util
