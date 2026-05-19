#include <inferrt/util/Path.hpp>

#include <vector>

namespace irt::util {

std::filesystem::path findProjectRoot(const char                                  *program_name,
                                      std::initializer_list<std::filesystem::path> required_paths,
                                      const char                                  *source_file)
{
    namespace fs = std::filesystem;

    std::vector<fs::path> starts;
    if (source_file && *source_file)
    {
        starts.push_back(fs::path(source_file).parent_path());
    }
    starts.push_back(fs::current_path());
    if (program_name && *program_name)
    {
        starts.push_back(fs::absolute(program_name).parent_path());
    }

    for (auto start : starts)
    {
        for (fs::path path = fs::absolute(start); !path.empty(); path = path.parent_path())
        {
            bool found = true;
            for (const auto &required_path : required_paths)
            {
                if (!fs::exists(path / required_path))
                {
                    found = false;
                    break;
                }
            }
            if (found)
            {
                return path;
            }
            if (path == path.root_path())
            {
                break;
            }
        }
    }

    return fs::current_path();
}

} // namespace irt::util
