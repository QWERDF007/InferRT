#include <cuda_runtime_api.h>
#include <dlpack/dlpack.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/cvcuda/OpNMS.h>
#include <inferrt/cvcuda/OpRoIAlign.h>
#include <inferrt/ops/NMS.hpp>
#include <inferrt/ops/RoIAlign.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

struct TensorView
{
    py::object           object;
    py::object           dlpack_capsule;
    DLManagedTensor     *managed_tensor{nullptr};
    std::vector<int64_t> shape;
    void                *data{nullptr};

    [[nodiscard]] const DLTensor &tensor() const
    {
        return managed_tensor->dl_tensor;
    }
};

bool hasDLPackProtocol(const py::handle &object)
{
    return PyObject_HasAttrString(object.ptr(), "__dlpack__") == 1;
}

py::object callDLPack(const py::handle &object, uintptr_t stream_ptr)
{
    py::object method = py::reinterpret_steal<py::object>(PyObject_GetAttrString(object.ptr(), "__dlpack__"));
    if (!method)
    {
        throw py::error_already_set();
    }

    py::tuple args(0);
    py::dict  kwargs;
    if (stream_ptr != 0)
    {
        kwargs[py::str("stream")] = py::int_(stream_ptr);
    }

    PyObject *result = PyObject_Call(method.ptr(), args.ptr(), kwargs.ptr());
    if (result == nullptr)
    {
        throw py::error_already_set();
    }
    return py::reinterpret_steal<py::object>(result);
}

DLManagedTensor *dlpackCapsulePointer(const py::handle &capsule)
{
    if (!PyCapsule_CheckExact(capsule.ptr()))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "DLPack provider must return a PyCapsule from __dlpack__()");
    }

    auto *managed_tensor = static_cast<DLManagedTensor *>(PyCapsule_GetPointer(capsule.ptr(), "dltensor"));
    if (managed_tensor == nullptr)
    {
        throw py::error_already_set();
    }
    return managed_tensor;
}

std::vector<int64_t> dlShapeToVector(const DLTensor &tensor)
{
    if (tensor.ndim < 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "DLPack tensor rank is invalid");
    }
    if (tensor.ndim > 0 && tensor.shape == nullptr)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "DLPack tensor shape pointer is null");
    }

    std::vector<int64_t> shape;
    shape.reserve(static_cast<size_t>(tensor.ndim));
    for (int32_t i = 0; i < tensor.ndim; ++i)
    {
        if (tensor.shape[i] < 0)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "DLPack tensor has negative dimension at %d", i);
        }
        shape.push_back(tensor.shape[i]);
    }
    return shape;
}

bool isCompactRowMajor(const DLTensor &tensor)
{
    if (tensor.strides == nullptr || tensor.ndim <= 1)
    {
        return true;
    }

    int64_t expected_stride = 1;
    for (int32_t i = tensor.ndim - 1; i >= 0; --i)
    {
        if (tensor.shape[i] == 0)
        {
            return true;
        }
        if (tensor.shape[i] != 1 && tensor.strides[i] != expected_stride)
        {
            return false;
        }
        expected_stride *= tensor.shape[i];
    }
    return true;
}

bool isFloat32(const DLDataType &dtype)
{
    return dtype.code == kDLFloat && dtype.bits == 32 && dtype.lanes == 1;
}

bool dlDeviceIsHost(const DLDevice &device)
{
    return device.device_type == kDLCPU || device.device_type == kDLCUDAHost;
}

bool dlDeviceIsCuda(const DLDevice &device)
{
    return device.device_type == kDLCUDA || device.device_type == kDLCUDAManaged;
}

int currentCudaDevice()
{
    int device = 0;
    const cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess)
    {
        throw irt::Exception(irt::Status::ERROR_DEVICE, "Failed to query current CUDA device: %s",
                             cudaGetErrorString(err));
    }
    return device;
}

TensorView makeTensorView(const py::handle &object, uintptr_t stream_ptr, const char *name)
{
    if (!hasDLPackProtocol(object))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "%s must implement __dlpack__()", name);
    }

    TensorView view;
    view.object         = py::reinterpret_borrow<py::object>(object);
    view.dlpack_capsule = callDLPack(object, stream_ptr);
    view.managed_tensor = dlpackCapsulePointer(view.dlpack_capsule);

    const auto &tensor = view.tensor();
    view.shape         = dlShapeToVector(tensor);
    if (!isCompactRowMajor(tensor))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "%s must be contiguous row-major", name);
    }

    auto *base_ptr = static_cast<char *>(tensor.data);
    if (base_ptr == nullptr && !view.shape.empty()
        && std::accumulate(view.shape.begin(), view.shape.end(), int64_t{1}, std::multiplies<int64_t>()) > 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "%s data pointer is null", name);
    }
    view.data = base_ptr == nullptr ? nullptr : base_ptr + tensor.byte_offset;

    if (dlDeviceIsCuda(tensor.device) && tensor.device.device_id != currentCudaDevice())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "%s CUDA device mismatch: expected current device %d, got %d", name, currentCudaDevice(),
                             tensor.device.device_id);
    }
    if (!dlDeviceIsHost(tensor.device) && !dlDeviceIsCuda(tensor.device))
    {
        throw irt::Exception(irt::Status::ERROR_NOT_IMPLEMENTED, "%s has unsupported DLPack device type=%d", name,
                             static_cast<int>(tensor.device.device_type));
    }

    return view;
}

void validateFloat32(const TensorView &view, const char *name)
{
    if (!isFloat32(view.tensor().dtype))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "%s must be float32", name);
    }
}

void validateRank(const TensorView &view, int rank, const char *name)
{
    if (static_cast<int>(view.shape.size()) != rank)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "%s rank mismatch: expected=%d got=%zu", name, rank,
                             view.shape.size());
    }
}

void ensureSameDevice(const TensorView &lhs, const TensorView &rhs, const char *lhs_name, const char *rhs_name)
{
    const auto &lhs_device = lhs.tensor().device;
    const auto &rhs_device = rhs.tensor().device;
    if (dlDeviceIsHost(lhs_device) && dlDeviceIsHost(rhs_device))
    {
        return;
    }
    if (lhs_device.device_type == rhs_device.device_type && lhs_device.device_id == rhs_device.device_id)
    {
        return;
    }

    throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                         "%s and %s must be on the same device, got type/id=(%d,%d) and (%d,%d)", lhs_name, rhs_name,
                         static_cast<int>(lhs_device.device_type), lhs_device.device_id,
                         static_cast<int>(rhs_device.device_type), rhs_device.device_id);
}

int checkedDimToInt(int64_t value, const char *name)
{
    if (value < 0 || value > std::numeric_limits<int>::max())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "%s is out of int range: %lld", name,
                             static_cast<long long>(value));
    }
    return static_cast<int>(value);
}

std::string torchDeviceString(const DLDevice &device)
{
    if (dlDeviceIsHost(device))
    {
        return "cpu";
    }
    if (dlDeviceIsCuda(device))
    {
        return "cuda:" + std::to_string(device.device_id);
    }
    throw irt::Exception(irt::Status::ERROR_NOT_IMPLEMENTED, "Unsupported output device type=%d",
                         static_cast<int>(device.device_type));
}

py::tuple shapeTuple(const std::vector<int64_t> &shape)
{
    py::tuple tuple(static_cast<py::ssize_t>(shape.size()));
    for (size_t i = 0; i < shape.size(); ++i)
    {
        tuple[static_cast<py::ssize_t>(i)] = py::int_(shape[i]);
    }
    return tuple;
}

py::object torchEmpty(const std::vector<int64_t> &shape, const char *dtype_attr, const DLDevice &device)
{
    py::module_ torch = py::module_::import("torch");
    py::dict    kwargs;
    kwargs[py::str("dtype")]  = torch.attr(dtype_attr);
    kwargs[py::str("device")] = py::str(torchDeviceString(device));

    py::tuple args = py::make_tuple(shapeTuple(shape));
    PyObject *result = PyObject_Call(torch.attr("empty").ptr(), args.ptr(), kwargs.ptr());
    if (result == nullptr)
    {
        throw py::error_already_set();
    }
    return py::reinterpret_steal<py::object>(result);
}

void *torchDataPtr(const py::object &tensor, const char *name)
{
    const uintptr_t ptr = tensor.attr("data_ptr")().cast<uintptr_t>();
    if (ptr == 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "%s data_ptr is null", name);
    }
    return reinterpret_cast<void *>(ptr);
}

void throwIfStatus(IRTStatus status, const char *op_name)
{
    if (status == IRT_SUCCESS)
    {
        return;
    }

    char msg[IRT_MAX_STATUS_MESSAGE_LENGTH] = {};
    irt::PeekAtLastErrorMessage(msg, sizeof(msg));
    throw irt::Exception(static_cast<irt::Status>(status), "%s failed: %s", op_name, msg);
}

void checkCuda(cudaError_t err, const char *message)
{
    if (err != cudaSuccess)
    {
        throw irt::Exception(irt::Status::ERROR_DEVICE, "%s: %s", message, cudaGetErrorString(err));
    }
}

std::pair<int, int> parseOutputSize(const py::object &value)
{
    if (py::isinstance<py::int_>(value))
    {
        const int size = value.cast<int>();
        return {size, size};
    }

    const auto values = value.cast<std::vector<int>>();
    if (values.size() != 2)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "output_size must be an int or a sequence of two ints");
    }
    return {values[0], values[1]};
}

py::array_t<float, py::array::c_style | py::array::forcecast> asFloat32CArray(const py::handle &value, const char *name)
{
    auto                                                          object = py::reinterpret_borrow<py::object>(value);
    py::array_t<float, py::array::c_style | py::array::forcecast> array(object);
    if (!array)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "%s must be convertible to float32 ndarray", name);
    }
    return array;
}

std::array<int64_t, 4> inputShape(const py::array_t<float, py::array::c_style | py::array::forcecast> &input)
{
    if (input.ndim() != 4)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "input must have shape [N, C, H, W]");
    }
    return {input.shape(0), input.shape(1), input.shape(2), input.shape(3)};
}

int64_t numRois(const py::array_t<float, py::array::c_style | py::array::forcecast> &rois)
{
    if (rois.ndim() != 2 || rois.shape(1) != 5)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "rois must have shape [K, 5]");
    }
    return rois.shape(0);
}

int64_t numBoxes(const py::array_t<float, py::array::c_style | py::array::forcecast> &boxes)
{
    if (boxes.ndim() != 2 || boxes.shape(1) != 4)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "boxes must have shape [N, 4]");
    }
    return boxes.shape(0);
}

void validateScores(const py::array_t<float, py::array::c_style | py::array::forcecast> &scores, int64_t expected)
{
    if (scores.ndim() != 1)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "scores must have shape [N]");
    }
    if (scores.shape(0) != expected)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "boxes and scores must contain the same number of elements");
    }
}

py::array_t<float> runRoIAlign(const irt::ops::RoIAlign &roi_align, const py::handle &input_object,
                               const py::handle &rois_object)
{
    auto input = asFloat32CArray(input_object, "input");
    auto rois  = asFloat32CArray(rois_object, "rois");

    const auto shape = inputShape(input);
    const auto k     = numRois(rois);

    py::array_t<float> output(
        {k, shape[1], static_cast<int64_t>(roi_align.pooledHeight()), static_cast<int64_t>(roi_align.pooledWidth())});

    {
        py::gil_scoped_release release;
        roi_align.forward(input.data(), shape.data(), rois.data(), k, output.mutable_data());
    }

    return output;
}

py::array_t<int64_t> runNMS(const py::handle &boxes_object, const py::handle &scores_object, float iou_threshold)
{
    auto boxes  = asFloat32CArray(boxes_object, "boxes");
    auto scores = asFloat32CArray(scores_object, "scores");

    const auto n = numBoxes(boxes);
    validateScores(scores, n);

    std::vector<int64_t> keep;
    {
        py::gil_scoped_release release;
        keep = irt::ops::nms(boxes.data(), scores.data(), n, iou_threshold);
    }

    py::array_t<int64_t> output(static_cast<py::ssize_t>(keep.size()));
    std::copy(keep.begin(), keep.end(), output.mutable_data());
    return output;
}

py::object runNMSV2(const py::handle &boxes_object, const py::handle &scores_object, float iou_threshold,
                    uintptr_t stream_ptr)
{
    auto boxes  = makeTensorView(boxes_object, stream_ptr, "boxes");
    auto scores = makeTensorView(scores_object, stream_ptr, "scores");

    validateFloat32(boxes, "boxes");
    validateFloat32(scores, "scores");
    validateRank(boxes, 2, "boxes");
    validateRank(scores, 1, "scores");
    if (boxes.shape[1] != 4)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "boxes must have shape [N, 4]");
    }
    if (scores.shape[0] != boxes.shape[0])
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "boxes and scores must contain the same number of elements");
    }
    ensureSameDevice(boxes, scores, "boxes", "scores");

    const int num_boxes = checkedDimToInt(boxes.shape[0], "num_boxes");
    py::object output   = torchEmpty({boxes.shape[0]}, "int64", boxes.tensor().device);
    if (num_boxes == 0)
    {
        return output;
    }

    auto *out_ptr = static_cast<int64_t *>(torchDataPtr(output, "output"));
    if (dlDeviceIsCuda(boxes.tensor().device))
    {
        py::object count_tensor = torchEmpty({1}, "int32", boxes.tensor().device);
        auto      *count_ptr    = static_cast<int *>(torchDataPtr(count_tensor, "keep_count"));
        const auto stream       = reinterpret_cast<cudaStream_t>(stream_ptr);

        int keep_count = 0;
        {
            py::gil_scoped_release release;
            const auto status = irt::cvcuda::nms(static_cast<const float *>(boxes.data),
                                                 static_cast<const float *>(scores.data), out_ptr, count_ptr,
                                                 num_boxes, iou_threshold, stream);
            throwIfStatus(status, "cvcuda.nms");
            checkCuda(cudaMemcpyAsync(&keep_count, count_ptr, sizeof(int), cudaMemcpyDeviceToHost, stream),
                      "Failed to copy NMS keep count to host");
            checkCuda(cudaStreamSynchronize(stream), "Failed to synchronize NMS stream");
        }
        return output.attr("narrow")(0, 0, keep_count);
    }

    int64_t keep_size = 0;
    {
        py::gil_scoped_release release;
        const auto keep
            = irt::ops::nms(static_cast<const float *>(boxes.data), static_cast<const float *>(scores.data), num_boxes,
                            iou_threshold);
        std::copy(keep.begin(), keep.end(), out_ptr);
        keep_size = static_cast<int64_t>(keep.size());
    }
    return output.attr("narrow")(0, 0, keep_size);
}

py::object runRoIAlignV2(const py::handle &input_object, const py::handle &rois_object, const py::object &output_size,
                         float spatial_scale, int sampling_ratio, bool aligned, uintptr_t stream_ptr)
{
    auto input = makeTensorView(input_object, stream_ptr, "input");
    auto rois  = makeTensorView(rois_object, stream_ptr, "rois");

    validateFloat32(input, "input");
    validateFloat32(rois, "rois");
    validateRank(input, 4, "input");
    validateRank(rois, 2, "rois");
    if (rois.shape[1] != 5)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "rois must have shape [K, 5]");
    }
    ensureSameDevice(input, rois, "input", "rois");

    const auto parsed_output_size = parseOutputSize(output_size);
    const int  pooled_height      = parsed_output_size.first;
    const int  pooled_width       = parsed_output_size.second;
    if (pooled_height <= 0 || pooled_width <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "output_size must be positive");
    }

    const int batches  = checkedDimToInt(input.shape[0], "input.shape[0]");
    const int channels = checkedDimToInt(input.shape[1], "input.shape[1]");
    const int height   = checkedDimToInt(input.shape[2], "input.shape[2]");
    const int width    = checkedDimToInt(input.shape[3], "input.shape[3]");
    const int num_rois = checkedDimToInt(rois.shape[0], "rois.shape[0]");

    py::object output = torchEmpty({rois.shape[0], input.shape[1], pooled_height, pooled_width}, "float32",
                                   input.tensor().device);
    if (num_rois == 0)
    {
        return output;
    }

    auto *out_ptr = static_cast<float *>(torchDataPtr(output, "output"));
    if (dlDeviceIsCuda(input.tensor().device))
    {
        const auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
        {
            py::gil_scoped_release release;
            const auto status = irt::cvcuda::roiAlign(static_cast<const float *>(input.data),
                                                      static_cast<const float *>(rois.data), out_ptr, batches, channels,
                                                      cv::Size(width, height), num_rois,
                                                      cv::Size(pooled_width, pooled_height), spatial_scale,
                                                      sampling_ratio, aligned, stream);
            throwIfStatus(status, "cvcuda.roiAlign");
        }
        return output;
    }

    {
        py::gil_scoped_release release;
        const int64_t input_shape[4] = {batches, channels, height, width};
        const irt::ops::RoIAlign op({pooled_height, pooled_width}, spatial_scale, sampling_ratio, aligned);
        op.forward(static_cast<const float *>(input.data), input_shape, static_cast<const float *>(rois.data),
                   num_rois, out_ptr);
    }
    return output;
}

std::string reprRoIAlign(const irt::ops::RoIAlign &self)
{
    return "RoIAlign(output_size=(" + std::to_string(self.pooledHeight()) + ", " + std::to_string(self.pooledWidth())
         + "), spatial_scale=" + std::to_string(self.spatialScale()) + ", sampling_ratio="
         + std::to_string(self.samplingRatio()) + ", aligned=" + (self.aligned() ? "True" : "False") + ")";
}

} // namespace

PYBIND11_MODULE(inferrt_ops_py, m)
{
    m.doc() = "InferRT ops Python bindings";

    py::register_exception<irt::Exception>(m, "InferRTOpsError");

    py::class_<irt::ops::RoIAlign>(m, "RoIAlign")
        .def(py::init<int, int, float, int, bool>(), py::arg("pooled_height"), py::arg("pooled_width"),
             py::arg("spatial_scale") = 1.0f, py::arg("sampling_ratio") = -1, py::arg("aligned") = false)
        .def(py::init(
                 [](const py::object &output_size, float spatial_scale, int sampling_ratio, bool aligned)
                 { return irt::ops::RoIAlign(parseOutputSize(output_size), spatial_scale, sampling_ratio, aligned); }),
             py::arg("output_size"), py::arg("spatial_scale") = 1.0f, py::arg("sampling_ratio") = -1,
             py::arg("aligned") = false)
        .def_property_readonly("output_size", [](const irt::ops::RoIAlign &self)
                               { return std::vector<int>{self.pooledHeight(), self.pooledWidth()}; })
        .def_property_readonly("spatial_scale", &irt::ops::RoIAlign::spatialScale)
        .def_property_readonly("sampling_ratio", &irt::ops::RoIAlign::samplingRatio)
        .def_property_readonly("aligned", &irt::ops::RoIAlign::aligned)
        .def("forward", &runRoIAlign, py::arg("input"), py::arg("rois"))
        .def("__call__", &runRoIAlign, py::arg("input"), py::arg("rois"))
        .def("__repr__", &reprRoIAlign);

    m.def(
        "roi_align",
        [](const py::handle &input, const py::handle &rois, const py::object &output_size, float spatial_scale,
           int sampling_ratio, bool aligned)
        {
            const irt::ops::RoIAlign op(parseOutputSize(output_size), spatial_scale, sampling_ratio, aligned);
            return runRoIAlign(op, input, rois);
        },
        py::arg("input"), py::arg("rois"), py::arg("output_size"), py::arg("spatial_scale") = 1.0f,
        py::arg("sampling_ratio") = -1, py::arg("aligned") = false);

    m.def("nms", &runNMS, py::arg("boxes"), py::arg("scores"), py::arg("iou_threshold"),
          "Performs non-maximum suppression on boxes in xyxy format.");

    m.def("nms_v2", &runNMSV2, py::arg("boxes"), py::arg("scores"), py::arg("iou_threshold"),
          py::arg("stream_ptr") = uintptr_t{0},
          "DLPack tensor NMS. CPU tensors use InferRT CPU ops; CUDA tensors use InferRT cvcuda zero-copy.");

    m.def("roi_align_v2", &runRoIAlignV2, py::arg("input"), py::arg("rois"), py::arg("output_size"),
          py::arg("spatial_scale") = 1.0f, py::arg("sampling_ratio") = -1, py::arg("aligned") = false,
          py::arg("stream_ptr") = uintptr_t{0},
          "DLPack tensor RoIAlign. CPU tensors use InferRT CPU ops; CUDA tensors use InferRT cvcuda zero-copy.");
}
