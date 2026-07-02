#include <cuda_runtime_api.h>
#include <dlpack/dlpack.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/cvcuda/OpNMS.h>
#include <inferrt/cvcuda/OpRoIAlign.h>
#include <inferrt/ops/BezierFit.hpp>
#include <inferrt/ops/BSplineInterp.hpp>
#include <inferrt/ops/DBSCAN.hpp>
#include <inferrt/ops/HDBSCAN.hpp>
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
#include <optional>
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
    int               device = 0;
    const cudaError_t err    = cudaGetDevice(&device);
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

    py::tuple args   = py::make_tuple(shapeTuple(shape));
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

std::pair<int64_t, int64_t> sampleMatrixShape(
    const py::array_t<float, py::array::c_style | py::array::forcecast> &samples)
{
    if (samples.ndim() != 2)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "samples must have shape [N, F]");
    }
    return {samples.shape(0), samples.shape(1)};
}

std::pair<int64_t, int64_t> pointMatrixShape(
    const py::array_t<float, py::array::c_style | py::array::forcecast> &points, const char *name)
{
    if (points.ndim() != 2)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "%s must have shape [N, D]", name);
    }
    return {points.shape(0), points.shape(1)};
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

template<typename Result, typename Config>
Result runSampleMatrixClustering(const py::handle &samples_object, const Config &config,
                                 Result (*op)(const float *, int64_t, int64_t, const Config &))
{
    auto samples                           = asFloat32CArray(samples_object, "samples");
    const auto [num_samples, num_features] = sampleMatrixShape(samples);

    py::gil_scoped_release release;
    return op(samples.data(), num_samples, num_features, config);
}

irt::ops::DBSCANResult runDBSCAN(const py::handle &samples_object, const irt::ops::DBSCANConfig &config)
{
    return runSampleMatrixClustering(samples_object, config, &irt::ops::dbscan);
}

irt::ops::HDBSCANResult runHDBSCAN(const py::handle &samples_object, const irt::ops::HDBSCANConfig &config)
{
    return runSampleMatrixClustering(samples_object, config, &irt::ops::hdbscan);
}

irt::ops::BezierFitResult runFitBezierCurve(const py::handle &points_object, int degree,
                                            const py::object &parameters_object)
{
    auto       points                 = asFloat32CArray(points_object, "points");
    const auto [num_points, num_dims] = pointMatrixShape(points, "points");

    std::optional<py::array_t<float, py::array::c_style | py::array::forcecast>> parameters;
    const float                                                              *parameter_data = nullptr;
    if (!parameters_object.is_none())
    {
        parameters.emplace(asFloat32CArray(parameters_object, "parameters"));
        if (parameters->ndim() != 1 || parameters->shape(0) != num_points)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "parameters must have shape [N] matching points");
        }
        parameter_data = parameters->data();
    }

    py::gil_scoped_release release;
    return irt::ops::fitBezierCurve(points.data(), num_points, num_dims, degree, parameter_data);
}

py::array_t<float> runEvaluateBezierCurve(const py::handle &control_points_object, const py::handle &parameters_object)
{
    auto       control_points                   = asFloat32CArray(control_points_object, "control_points");
    auto       parameters                       = asFloat32CArray(parameters_object, "parameters");
    const auto [num_control_points, num_dims]   = pointMatrixShape(control_points, "control_points");
    if (parameters.ndim() != 1)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "parameters must have shape [M]");
    }
    const auto num_parameters = parameters.shape(0);

    py::array_t<float> output({num_parameters, num_dims});
    {
        py::gil_scoped_release release;
        irt::ops::evaluateBezierCurve(control_points.data(), num_control_points, num_dims, parameters.data(),
                                      num_parameters, output.mutable_data());
    }
    return output;
}

irt::ops::BSplineInterpResult runMakeInterpSpline(const py::handle &x_object, const py::handle &y_object, int degree)
{
    auto x = asFloat32CArray(x_object, "x");
    auto y = asFloat32CArray(y_object, "y");
    if (x.ndim() != 1)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "x must have shape [N]");
    }
    const auto [num_points, num_dims] = pointMatrixShape(y, "y");
    if (x.shape(0) != num_points)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "x and y must contain the same number of points");
    }

    py::gil_scoped_release release;
    return irt::ops::makeInterpSpline(x.data(), y.data(), num_points, num_dims, degree);
}

irt::ops::SplPrepResult runSplPrep(const py::handle &points_object, float smoothing, int degree,
                                   const py::object &parameters_object)
{
    auto       points                 = asFloat32CArray(points_object, "points");
    const auto [num_points, num_dims] = pointMatrixShape(points, "points");

    std::optional<py::array_t<float, py::array::c_style | py::array::forcecast>> parameters;
    const float                                                              *parameter_data = nullptr;
    if (!parameters_object.is_none())
    {
        parameters.emplace(asFloat32CArray(parameters_object, "parameters"));
        if (parameters->ndim() != 1 || parameters->shape(0) != num_points)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "parameters must have shape [N] matching points");
        }
        parameter_data = parameters->data();
    }

    py::gil_scoped_release release;
    return irt::ops::splPrep(points.data(), num_points, num_dims, smoothing, degree, parameter_data);
}

py::array_t<float> runEvaluateBSpline(const py::handle &knots_object, const py::handle &coefficients_object,
                                      int degree, const py::handle &x_eval_object)
{
    auto knots        = asFloat32CArray(knots_object, "knots");
    auto coefficients = asFloat32CArray(coefficients_object, "coefficients");
    auto x_eval       = asFloat32CArray(x_eval_object, "x_eval");
    if (knots.ndim() != 1)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "knots must have shape [T]");
    }
    const auto [num_coefficients, num_dims] = pointMatrixShape(coefficients, "coefficients");
    if (x_eval.ndim() != 1)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "x_eval must have shape [M]");
    }

    py::array_t<float> output({x_eval.shape(0), num_dims});
    {
        py::gil_scoped_release release;
        irt::ops::evaluateBSpline(knots.data(), knots.shape(0), coefficients.data(), num_coefficients, num_dims,
                                  degree, x_eval.data(), x_eval.shape(0), output.mutable_data());
    }
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

    const int  num_boxes = checkedDimToInt(boxes.shape[0], "num_boxes");
    py::object output    = torchEmpty({boxes.shape[0]}, "int64", boxes.tensor().device);
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
            const auto             status
                = irt::cvcuda::nms(static_cast<const float *>(boxes.data), static_cast<const float *>(scores.data),
                                   out_ptr, count_ptr, num_boxes, iou_threshold, stream);
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
        const auto keep = irt::ops::nms(static_cast<const float *>(boxes.data), static_cast<const float *>(scores.data),
                                        num_boxes, iou_threshold);
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

    py::object output
        = torchEmpty({rois.shape[0], input.shape[1], pooled_height, pooled_width}, "float32", input.tensor().device);
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
            const auto             status = irt::cvcuda::roiAlign(
                static_cast<const float *>(input.data), static_cast<const float *>(rois.data), out_ptr, batches,
                channels, cv::Size(width, height), num_rois, cv::Size(pooled_width, pooled_height), spatial_scale,
                sampling_ratio, aligned, stream);
            throwIfStatus(status, "cvcuda.roiAlign");
        }
        return output;
    }

    {
        py::gil_scoped_release   release;
        const int64_t            input_shape[4] = {batches, channels, height, width};
        const irt::ops::RoIAlign op({pooled_height, pooled_width}, spatial_scale, sampling_ratio, aligned);
        op.forward(static_cast<const float *>(input.data), input_shape, static_cast<const float *>(rois.data), num_rois,
                   out_ptr);
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

    py::enum_<irt::ops::ClusteringAlgorithm>(m, "ClusteringAlgorithm")
        .value("Brute", irt::ops::ClusteringAlgorithm::Brute)
        .value("KDTree", irt::ops::ClusteringAlgorithm::KDTree)
        .value("BallTree", irt::ops::ClusteringAlgorithm::BallTree);

    py::enum_<irt::ops::ClusteringMetric>(m, "ClusteringMetric")
        .value("Euclidean", irt::ops::ClusteringMetric::Euclidean)
        .value("Cosine", irt::ops::ClusteringMetric::Cosine)
        .value("Manhattan", irt::ops::ClusteringMetric::Manhattan)
        .value("Chebyshev", irt::ops::ClusteringMetric::Chebyshev)
        .value("Minkowski", irt::ops::ClusteringMetric::Minkowski);

    py::class_<irt::ops::DBSCANConfig>(m, "DBSCANConfig")
        .def(py::init<>())
        .def_readwrite("eps", &irt::ops::DBSCANConfig::eps)
        .def_readwrite("min_samples", &irt::ops::DBSCANConfig::min_samples)
        .def_readwrite("algorithm", &irt::ops::DBSCANConfig::algorithm)
        .def_readwrite("leaf_size", &irt::ops::DBSCANConfig::leaf_size)
        .def_readwrite("metric", &irt::ops::DBSCANConfig::metric)
        .def_readwrite("minkowski_p", &irt::ops::DBSCANConfig::minkowski_p);

    py::class_<irt::ops::DBSCANResult>(m, "DBSCANResult")
        .def_readonly("core_sample_indices", &irt::ops::DBSCANResult::core_sample_indices)
        .def_readonly("labels", &irt::ops::DBSCANResult::labels);

    py::enum_<irt::ops::HDBSCANClusterSelectionMethod>(m, "HDBSCANClusterSelectionMethod")
        .value("Eom", irt::ops::HDBSCANClusterSelectionMethod::Eom)
        .value("Leaf", irt::ops::HDBSCANClusterSelectionMethod::Leaf);

    py::class_<irt::ops::HDBSCANConfig>(m, "HDBSCANConfig")
        .def(py::init<>())
        .def_readwrite("min_cluster_size", &irt::ops::HDBSCANConfig::min_cluster_size)
        .def_readwrite("min_samples", &irt::ops::HDBSCANConfig::min_samples)
        .def_readwrite("cluster_selection_epsilon", &irt::ops::HDBSCANConfig::cluster_selection_epsilon)
        .def_readwrite("max_cluster_size", &irt::ops::HDBSCANConfig::max_cluster_size)
        .def_readwrite("alpha", &irt::ops::HDBSCANConfig::alpha)
        .def_readwrite("algorithm", &irt::ops::HDBSCANConfig::algorithm)
        .def_readwrite("leaf_size", &irt::ops::HDBSCANConfig::leaf_size)
        .def_readwrite("metric", &irt::ops::HDBSCANConfig::metric)
        .def_readwrite("minkowski_p", &irt::ops::HDBSCANConfig::minkowski_p)
        .def_readwrite("cluster_selection_method", &irt::ops::HDBSCANConfig::cluster_selection_method)
        .def_readwrite("allow_single_cluster", &irt::ops::HDBSCANConfig::allow_single_cluster);

    py::class_<irt::ops::HDBSCANResult>(m, "HDBSCANResult")
        .def_readonly("labels", &irt::ops::HDBSCANResult::labels)
        .def_readonly("probabilities", &irt::ops::HDBSCANResult::probabilities);

    py::class_<irt::ops::BezierFitResult>(m, "BezierFitResult")
        .def_readonly("degree", &irt::ops::BezierFitResult::degree)
        .def_readonly("dimensions", &irt::ops::BezierFitResult::dimensions)
        .def_readonly("control_points", &irt::ops::BezierFitResult::control_points)
        .def_readonly("parameters", &irt::ops::BezierFitResult::parameters)
        .def_readonly("residual_sum_squares", &irt::ops::BezierFitResult::residual_sum_squares);

    py::class_<irt::ops::BSplineInterpResult>(m, "BSplineInterpResult")
        .def_readonly("degree", &irt::ops::BSplineInterpResult::degree)
        .def_readonly("dimensions", &irt::ops::BSplineInterpResult::dimensions)
        .def_readonly("knots", &irt::ops::BSplineInterpResult::knots)
        .def_readonly("coefficients", &irt::ops::BSplineInterpResult::coefficients);

    py::class_<irt::ops::SplPrepResult>(m, "SplPrepResult")
        .def_readonly("degree", &irt::ops::SplPrepResult::degree)
        .def_readonly("dimensions", &irt::ops::SplPrepResult::dimensions)
        .def_readonly("smoothing", &irt::ops::SplPrepResult::smoothing)
        .def_readonly("residual_sum_squares", &irt::ops::SplPrepResult::residual_sum_squares)
        .def_readonly("knots", &irt::ops::SplPrepResult::knots)
        .def_readonly("coefficients", &irt::ops::SplPrepResult::coefficients)
        .def_readonly("parameters", &irt::ops::SplPrepResult::parameters);

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

    m.def("dbscan", &runDBSCAN, py::arg("samples"), py::arg("config"),
          "Performs DBSCAN clustering on a float32 sample matrix using DBSCANConfig.");

    m.def("hdbscan", &runHDBSCAN, py::arg("samples"), py::arg("config"),
          "Performs HDBSCAN clustering on a float32 sample matrix using HDBSCANConfig.");

    m.def("fit_bezier_curve", &runFitBezierCurve, py::arg("points"), py::arg("degree"),
          py::arg("parameters") = py::none(),
          "Fits a single Bezier curve to [N, D] points using chord-length parameters by default.");

    m.def("evaluate_bezier_curve", &runEvaluateBezierCurve, py::arg("control_points"), py::arg("parameters"),
          "Evaluates Bezier control points at 1D parameters in [0, 1].");

    m.def("make_interp_spline", &runMakeInterpSpline, py::arg("x"), py::arg("y"), py::arg("degree") = 3,
          "Creates an interpolating B-spline for y[N, D], matching SciPy make_interp_spline for degree 1 and 3.");

    m.def("splprep", &runSplPrep, py::arg("points"), py::arg("smoothing") = 0.0F, py::arg("degree") = 3,
          py::arg("parameters") = py::none(),
          "Creates a parametric B-spline, matching SciPy splprep s=0 exactly and supporting residual-budget smoothing.");

    m.def("evaluate_b_spline", &runEvaluateBSpline, py::arg("knots"), py::arg("coefficients"), py::arg("degree"),
          py::arg("x_eval"), "Evaluates a B-spline returned by make_interp_spline.");

    m.def("nms_v2", &runNMSV2, py::arg("boxes"), py::arg("scores"), py::arg("iou_threshold"),
          py::arg("stream_ptr") = uintptr_t{0},
          "DLPack tensor NMS. CPU tensors use InferRT CPU ops; CUDA tensors use InferRT cvcuda zero-copy.");

    m.def("roi_align_v2", &runRoIAlignV2, py::arg("input"), py::arg("rois"), py::arg("output_size"),
          py::arg("spatial_scale") = 1.0f, py::arg("sampling_ratio") = -1, py::arg("aligned") = false,
          py::arg("stream_ptr") = uintptr_t{0},
          "DLPack tensor RoIAlign. CPU tensors use InferRT CPU ops; CUDA tensors use InferRT cvcuda zero-copy.");
}
