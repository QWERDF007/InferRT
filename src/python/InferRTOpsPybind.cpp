#include <inferrt/core/Exception.hpp>
#include <inferrt/ops/RoIAlign.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstdint>
#include <string>
#include <utility>

namespace py = pybind11;

namespace {

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
}
