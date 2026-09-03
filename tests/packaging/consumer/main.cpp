#if defined(_WIN32)
#    include <windows.h>
#endif

#include <inferrt/core/ModelContract.hpp>
#include <inferrt/core/Status.hpp>
#include <inferrt/core/Tensor.hpp>

int main()
{
    if (static_cast<IRTStatus>(irt::Status::INVALID_OPERATION) != IRT_ERROR_INVALID_OPERATION)
    {
        return 2;
    }
    irt::Shape shape{1, 3, 2, 2};
    return shape.elementCount() == 12 ? 0 : 1;
}
