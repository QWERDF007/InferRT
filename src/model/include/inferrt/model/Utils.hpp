#pragma once

#include <cuda_runtime_api.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/core/PreprocessSpec.hpp>
#include <inferrt/core/Tensor.hpp>
#include <inferrt/model/Export.h>
#include <opencv2/core.hpp>

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace irt::model {

/** Read the ImageNet label file used by the samples. */
INFERRT_MODEL_API std::vector<std::string> readImagenetLabels(const std::string &label_file);

/** Checked element count for a concrete core shape. */
INFERRT_MODEL_API size_t elementCount(const irt::Shape &shape);

/** Return the size of one core tensor element in bytes. */
INFERRT_MODEL_API size_t elementSize(irt::TensorDataType data_type);

/** Alias for elementSize kept at the core model boundary. */
INFERRT_MODEL_API size_t dataTypeSize(irt::TensorDataType data_type);

/** Return a stable, backend-neutral tensor type name. */
INFERRT_MODEL_API std::string dataTypeToString(irt::TensorDataType data_type);

/** Format a core shape as comma-separated dimensions. */
INFERRT_MODEL_API std::string dimsToCsv(const irt::Shape &shape);

/** Format a core shape for diagnostics. */
INFERRT_MODEL_API std::string dimsToString(const irt::Shape &shape);

/** Convert a CUDA status to an InferRT exception. */
INFERRT_MODEL_API void checkCuda(cudaError_t status, const char *op);

/** Select the CUDA device used by the current thread. */
INFERRT_MODEL_API void setCudaDevice(int device_id);

/** ImageNet preprocessing helpers shared by model samples and feature tools. */
class INFERRT_MODEL_API ImageNetUtil
{
public:
    using Geometry = irt::PreprocessGeometry;

    struct PreprocessResult
    {
        cv::Mat  image;
        Geometry geometry;
    };

    static const std::filesystem::path kDefaultImagePath;
    static const std::filesystem::path kDefaultLabelPath;

    /** Apply one backend-neutral image preprocessing specification. */
    static cv::Mat preprocess(const cv::Mat &image, const irt::PreprocessSpec &spec);

    /** Apply a spec and retain resize/letterbox geometry for postprocessing. */
    static PreprocessResult preprocessWithGeometry(const cv::Mat &image, const irt::PreprocessSpec &spec);

    /** Convenience ImageNet path; its defaults are derived from PreprocessSpec. */
    static cv::Mat preprocess(const cv::Mat &bgr_image, cv::Size target_size = cv::Size(224, 224));

    /** Convert a continuous or non-continuous HWC float image to channel-major data. */
    static std::vector<float> imageToTensorCHW(const cv::Mat &image);
};

} // namespace irt::model
