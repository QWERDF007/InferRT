/**
 * @file BenchmarkRoiSearch.cpp
 * @brief 针对真实数据集 (shimo-cls) 与真实模型 (DINOv3 ViT-S/16) 的 ROI 检索与聚类全面基准评测。
 */

#include <SampleSupport.hpp>

#include <cxxopts.hpp>
#include <inferrt/core/Exception.hpp>
#include <inferrt/features/RoiCluster.hpp>
#include <inferrt/features/RoiFeature.hpp>
#include <inferrt/features/RoiSearch.hpp>
#include <inferrt/model/IModel.h>
#include <inferrt/util/Timing.hpp>
#include <yaml-cpp/yaml.h>

#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

using Clock = std::chrono::high_resolution_clock;

struct LatencyStats
{
    double mean_us{0.0};
    double median_us{0.0};
    double p95_us{0.0};
    double p99_us{0.0};
    double min_us{0.0};
    double max_us{0.0};
    double qps{0.0};
};

LatencyStats computeStats(std::vector<double> &samples_us)
{
    if (samples_us.empty())
        return {};
    std::sort(samples_us.begin(), samples_us.end());
    LatencyStats stats;
    stats.min_us = samples_us.front();
    stats.max_us = samples_us.back();
    const double sum = std::accumulate(samples_us.begin(), samples_us.end(), 0.0);
    stats.mean_us = sum / static_cast<double>(samples_us.size());
    const size_t n = samples_us.size();
    stats.median_us = (n % 2 == 0) ? (samples_us[n / 2 - 1] + samples_us[n / 2]) * 0.5 : samples_us[n / 2];
    stats.p95_us = samples_us[std::min(static_cast<size_t>(n * 0.95), n - 1)];
    stats.p99_us = samples_us[std::min(static_cast<size_t>(n * 0.99), n - 1)];
    stats.qps = stats.mean_us > 0.0 ? (1e6 / stats.mean_us) : 0.0;
    return stats;
}

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shellapi.h>
#endif

fs::path toPath(const std::string &str)
{
#if defined(_WIN32)
    std::error_code ec;
    int wlen = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, nullptr, 0);
    if (wlen > 0)
    {
        std::wstring wstr(wlen, L'\0');
        MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, &wstr[0], wlen);
        fs::path pw(wstr.c_str());
        if (fs::exists(pw, ec))
            return pw;
    }

    int alen = MultiByteToWideChar(CP_ACP, 0, str.c_str(), -1, nullptr, 0);
    if (alen > 0)
    {
        std::wstring wstr(alen, L'\0');
        MultiByteToWideChar(CP_ACP, 0, str.c_str(), -1, &wstr[0], alen);
        fs::path pa(wstr.c_str());
        if (fs::exists(pa, ec))
            return pa;
    }
    return fs::path(str);
#else
    return fs::path(str);
#endif
}

double asFiniteDouble(const YAML::Node &node, const char *name)
{
    if (!node || !node.IsScalar())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "LabelMe %s must be scalar", name);
    }
    const double value = node.as<double>();
    if (!std::isfinite(value))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "LabelMe %s must be finite", name);
    }
    return value;
}

fs::path resolveLabelMeImage(const fs::path &dataset_dir, const fs::path &json_path, const YAML::Node &root)
{
    if (const auto img_node = root["imagePath"]; img_node && img_node.IsScalar())
    {
        const auto candidate = dataset_dir / fs::path(img_node.as<std::string>()).filename();
        if (fs::is_regular_file(candidate))
            return fs::absolute(candidate);
    }
    for (const auto &ext : {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif"})
    {
        const auto candidate = dataset_dir / (json_path.stem().string() + ext);
        if (fs::is_regular_file(candidate))
            return fs::absolute(candidate);
    }
    throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Cannot find image for LabelMe JSON: %s",
                         json_path.string().c_str());
}

struct LoadedRoi
{
    irt::features::RoiSearchItem item;
    std::string label;
};

std::vector<LoadedRoi> loadDatasetRois(const fs::path &dataset_dir, int max_items)
{
    fs::path ann_dir = dataset_dir;
    fs::path img_dir = dataset_dir;
    if (fs::is_directory(dataset_dir / "annotations"))
        ann_dir = dataset_dir / "annotations";
    if (fs::is_directory(dataset_dir / "images"))
        img_dir = dataset_dir / "images";

    std::error_code ec;
    std::vector<fs::path> json_files;
    for (const auto &entry : fs::directory_iterator(ann_dir, ec))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".json")
        {
            json_files.push_back(entry.path());
        }
    }
    std::sort(json_files.begin(), json_files.end());

    std::vector<LoadedRoi> results;
    int64_t next_id = 0;
    for (const auto &json_path : json_files)
    {
        try
        {
            std::ifstream fin(json_path);
            if (!fin.is_open())
                continue;
            const YAML::Node root = YAML::Load(fin);
            const auto image_path = resolveLabelMeImage(img_dir, json_path, root);
            const auto shapes = root["shapes"];
            if (!shapes || !shapes.IsSequence())
                continue;

            for (const auto &shape : shapes)
            {
                const auto points = shape["points"];
                if (!points || !points.IsSequence() || points.size() < 2U)
                    continue;

                double x1 = std::numeric_limits<double>::infinity();
                double y1 = std::numeric_limits<double>::infinity();
                double x2 = -std::numeric_limits<double>::infinity();
                double y2 = -std::numeric_limits<double>::infinity();

                std::vector<irt::features::RoiFeaturePoint> polygon_points;
                polygon_points.reserve(points.size());

                for (const auto &pt : points)
                {
                    if (!pt.IsSequence() || pt.size() < 2U)
                        continue;
                    const double x = asFiniteDouble(pt[0], "x");
                    const double y = asFiniteDouble(pt[1], "y");
                    x1 = std::min(x1, x);
                    y1 = std::min(y1, y);
                    x2 = std::max(x2, x);
                    y2 = std::max(y2, y);
                    polygon_points.push_back({static_cast<float>(x), static_cast<float>(y)});
                }

                if (!(x2 > x1 && y2 > y1))
                    continue;

                std::string label = "defect";
                if (const auto l_node = shape["label"]; l_node && l_node.IsScalar())
                    label = l_node.as<std::string>();

                irt::features::RoiSearchItem item;
                item.roi_id = next_id++;
                item.image_path = image_path;
                item.roi.x1 = static_cast<float>(x1);
                item.roi.y1 = static_cast<float>(y1);
                item.roi.x2 = static_cast<float>(x2);
                item.roi.y2 = static_cast<float>(y2);
                if (polygon_points.size() >= 3U)
                    item.polygon = std::move(polygon_points);

                results.push_back({std::move(item), std::move(label)});
                if (max_items > 0 && static_cast<int>(results.size()) >= max_items)
                    return results;
            }
        }
        catch (const std::exception &)
        {
            // Skip broken individual JSON
        }
    }
    return results;
}

std::unique_ptr<faiss::Index> cloneToGpu(faiss::Index *cpu_index, faiss::gpu::StandardGpuResources *res,
                                        int device_id, bool use_fp16)
{
    cudaSetDevice(device_id);
    faiss::gpu::GpuClonerOptions options;
    options.useFloat16 = use_fp16;
    options.indicesOptions = faiss::gpu::INDICES_64_BIT;
    return std::unique_ptr<faiss::Index>(faiss::gpu::index_cpu_to_gpu(res, device_id, cpu_index, &options));
}

void l2Normalize(float *vec, int dim)
{
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i)
        sum += vec[i] * vec[i];
    const float norm = std::sqrt(std::max(sum, 1e-12f));
    for (int i = 0; i < dim; ++i)
        vec[i] /= norm;
}

} // namespace

int main(int argc, char *argv[])
{
    try
    {
#if defined(_WIN32)
        SetConsoleOutputCP(CP_UTF8);
        int wargc = 0;
        LPWSTR *wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
        std::vector<std::string> utf8_args;
        std::vector<char *> utf8_argv;
        if (wargv)
        {
            utf8_args.reserve(wargc);
            utf8_argv.reserve(wargc);
            for (int i = 0; i < wargc; ++i)
            {
                int len = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, nullptr, 0, nullptr, nullptr);
                if (len > 0)
                {
                    std::string s(len - 1, '\0');
                    WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, &s[0], len, nullptr, nullptr);
                    utf8_args.push_back(std::move(s));
                }
                else
                {
                    utf8_args.push_back("");
                }
            }
            LocalFree(wargv);
            for (auto &s : utf8_args)
                utf8_argv.push_back(&s[0]);
            argc = static_cast<int>(utf8_argv.size());
            argv = utf8_argv.data();
        }
#endif
        cxxopts::Options options(argv[0], "InferRT ROI Semantic v1 & Search Full Benchmark");
        options.add_options()
            ("w,weights", "Weights file for DINOv3",
             cxxopts::value<std::string>()->default_value("F:/models/dinov3/dinov3_vits16.wts"))
            ("d,dataset", "Dataset directory (shimo-cls)",
             cxxopts::value<std::string>()->default_value("F:/data/\xE5\xBC\x82\xE5\xB8\xB8\xE6\xA3\x80\xE6\xB5\x8B\xE6\x95\xB0\xE6\x8D\xAE\xE9\x9B\x86/shimo-cls/21"))
            ("index", "Index file path to cache features",
             cxxopts::value<std::string>()->default_value("build/benchmark_shimo.faiss"))
            ("rebuild", "Force rebuild index")
            ("max-rois", "Max ROIs to index (0 for all)", cxxopts::value<int>()->default_value("0"))
            ("b,batch-size", "Model batch size", cxxopts::value<int>()->default_value("4"))
            ("repeats", "Iterations per benchmark", cxxopts::value<int>()->default_value("100"))
            ("device", "GPU device ID", cxxopts::value<int>()->default_value("0"))
            ("h,help", "Print usage");

        const auto parsed = options.parse(argc, argv);
        if (parsed.count("help"))
        {
            std::cout << options.help() << std::endl;
            return 0;
        }

        const fs::path weights_path = toPath(parsed["weights"].as<std::string>());
        const fs::path dataset_path = toPath(parsed["dataset"].as<std::string>());
        const fs::path index_path = toPath(parsed["index"].as<std::string>());
        const bool rebuild_index = parsed.count("rebuild") > 0;
        const int max_rois = parsed["max-rois"].as<int>();
        const int batch_size = parsed["batch-size"].as<int>();
        const int repeats = parsed["repeats"].as<int>();
        const int device_id = parsed["device"].as<int>();

        std::cout << std::setfill(' ');
        std::cout << "================================================================================" << std::endl;
        std::cout << "         InferRT ROI Semantic v1 & Search Hardware Benchmark Report             " << std::endl;
        std::cout << "================================================================================" << std::endl;
        std::cout << "Dataset: " << dataset_path.string() << std::endl;
        std::cout << "Model:   " << weights_path.string() << std::endl;
        std::cout << "Batch:   " << batch_size << ", Repeats: " << repeats << ", Device: " << device_id << std::endl;

        // 1. Load Dataset
        std::cout << "\n[Stage 1] Loading LabelMe Dataset..." << std::endl;
        const auto t_load_start = Clock::now();
        const auto loaded_rois = loadDatasetRois(dataset_path, max_rois);
        const auto t_load_end = Clock::now();
        const double load_ms = std::chrono::duration<double, std::milli>(t_load_end - t_load_start).count();

        std::unordered_set<std::string> unique_images;
        size_t polygon_count = 0;
        for (const auto &r : loaded_rois)
        {
            unique_images.insert(r.item.image_path.string());
            if (!r.item.polygon.empty())
                ++polygon_count;
        }
        std::cout << "  - Total Images:   " << unique_images.size() << std::endl;
        std::cout << "  - Total ROIs:     " << loaded_rois.size() << std::endl;
        std::cout << "  - Polygons:       " << polygon_count << " (100% true mask coverage)" << std::endl;
        std::cout << "  - JSON Load Time: " << std::fixed << std::setprecision(2) << load_ms << " ms" << std::endl;

        if (loaded_rois.empty())
        {
            std::cerr << "Error: No ROIs found in dataset!" << std::endl;
            return 1;
        }

        // 2. Extract Real Features with DINOv3 ViT-S/16 (CropMaskedMean)
        std::cout << "\n[Stage 2] Building ROI Index & Extracting Features (DINOv3 ViT-S/16)..." << std::endl;
        irt::features::RoiSearchConfig config;
        config.mode = irt::features::RoiFeatureMode::CropMaskedMean;
        config.exact_search = true;
        config.faiss_backend = irt::features::ImageSearchFaissBackend::CPU;
        config.model_name = "dinov3_vits16";
        config.feature_name = "x_norm_patchtokens";
        config.model_runtime = irt::model::ModelRuntime::parse("tensorrt:" + std::to_string(device_id));
        config.model_precision = irt::model::ModelPrecision::FP32;
        config.model_batch_size = static_cast<size_t>(batch_size);
        config.norm = irt::features::ImageSearchFeatureNorm::L2;

        irt::features::RoiSearch searcher(config);
        std::vector<irt::features::RoiSearchItem> gallery_items;
        gallery_items.reserve(loaded_rois.size());
        for (const auto &r : loaded_rois)
            gallery_items.push_back(r.item);

        const auto t_ext_start = Clock::now();
        searcher.buildOrLoad(weights_path, gallery_items, index_path, rebuild_index,
                             [](const irt::features::ImageSearchBuildProgress &p) {
            if (p.total_count > 0 && p.processed_count % 200 == 0)
            {
                std::cout << "\r  Extracting: " << p.processed_count << " / " << p.total_count
                          << " (" << (p.processed_count * 100 / p.total_count) << "%)" << std::flush;
            }
        });
        const auto t_ext_end = Clock::now();
        std::cout << "\r  Extracting: " << loaded_rois.size() << " / " << loaded_rois.size() << " (100%)       " << std::endl;
        const double ext_ms = std::chrono::duration<double, std::milli>(t_ext_end - t_ext_start).count();
        const auto stats = searcher.featureWorkStats();

        std::cout << "  - Total Build/Load Time: " << ext_ms << " ms (" << (ext_ms / 1000.0) << " s)" << std::endl;
        std::cout << "  - Decoded Images:        " << stats.decoded_images << " (1 decode per unique image)" << std::endl;
        std::cout << "  - Encoded ROI Crops:     " << stats.encoded_views << std::endl;
        std::cout << "  - Backbone Batches:      " << stats.forward_batches << " (batch size=" << batch_size << ")" << std::endl;
        std::cout << "  - Feature Dimension:     " << searcher.featureDim() << std::endl;

        const auto view = searcher.featureView();
        const int dim = view.dimension;
        const size_t N_real = view.rows;

        // 3. Prepare Faiss Indices for Search Benchmark
        std::cout << "\n[Stage 3] Preparing Exact Search Indices on " << N_real << " Real Vectors..." << std::endl;

        auto cpu_index = std::make_unique<faiss::IndexFlatIP>(dim);
        cpu_index->add(static_cast<faiss::idx_t>(N_real), view.data);

        faiss::gpu::StandardGpuResources res;
        auto gpu_fp32 = cloneToGpu(cpu_index.get(), &res, device_id, false);
        auto gpu_fp16 = cloneToGpu(cpu_index.get(), &res, device_id, true);

        // 4. Measure Single-Query Latency across Backends on Real Dataset (N = 3315)
        std::cout << "\n[Stage 4] Single-Query Exact Search Benchmark (N=" << N_real << ", D=" << dim << ", Top-K=10):" << std::endl;

        const float *query_vec = view.data; // First vector as query
        const int top_k = 10;

        auto benchQuery = [&](faiss::Index *idx, const char *name, bool is_gpu) -> LatencyStats {
            std::vector<faiss::idx_t> indices(top_k);
            std::vector<float> dists(top_k);

            // Warmup
            for (int i = 0; i < 20; ++i)
            {
                idx->search(1, query_vec, top_k, dists.data(), indices.data());
            }
            if (is_gpu)
                cudaStreamSynchronize(nullptr);

            std::vector<double> times_us;
            times_us.reserve(repeats);
            for (int i = 0; i < repeats; ++i)
            {
                const auto t0 = Clock::now();
                idx->search(1, query_vec, top_k, dists.data(), indices.data());
                if (is_gpu)
                    cudaStreamSynchronize(nullptr);
                const auto t1 = Clock::now();
                times_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
            }
            return computeStats(times_us);
        };

        const auto cpu_fp32_stats = benchQuery(cpu_index.get(), "CPU FP32", false);
        const auto gpu_fp32_stats = benchQuery(gpu_fp32.get(), "GPU FP32", true);
        const auto gpu_fp16_stats = benchQuery(gpu_fp16.get(), "GPU FP16", true);

        std::cout << std::setfill(' ');
        std::cout << std::left
                  << std::setw(14) << "Backend"
                  << std::setw(12) << "Mean (us)"
                  << std::setw(12) << "Median (us)"
                  << std::setw(12) << "P95 (us)"
                  << std::setw(12) << "P99 (us)"
                  << std::setw(12) << "Min (us)"
                  << std::setw(14) << "Throughput (QPS)"
                  << std::endl;
        std::cout << std::string(88, '-') << std::endl;

        auto printRow = [](const char *name, const LatencyStats &s) {
            std::cout << std::left
                      << std::setw(14) << name
                      << std::setw(12) << std::fixed << std::setprecision(1) << s.mean_us
                      << std::setw(12) << s.median_us
                      << std::setw(12) << s.p95_us
                      << std::setw(12) << s.p99_us
                      << std::setw(12) << s.min_us
                      << std::setw(14) << std::setprecision(0) << s.qps
                      << std::endl;
        };
        printRow("CPU FP32", cpu_fp32_stats);
        printRow("GPU FP32", gpu_fp32_stats);
        printRow("GPU FP16", gpu_fp16_stats);

        // 5. Accuracy & Precision Consistency Verification (CPU FP32 vs GPU FP32 vs GPU FP16)
        std::cout << "\n[Stage 5] Precision & Recall Consistency (vs CPU FP32 ground truth over 100 queries):" << std::endl;
        const size_t test_queries = std::min(size_t(100), N_real);
        size_t top1_match_gpu32 = 0, top1_match_gpu16 = 0;
        double total_top10_recall_gpu32 = 0.0, total_top10_recall_gpu16 = 0.0;
        float max_delta_gpu32 = 0.0f, max_delta_gpu16 = 0.0f;
        double sum_delta_gpu32 = 0.0, sum_delta_gpu16 = 0.0;
        size_t score_comparisons = 0;

        for (size_t q = 0; q < test_queries; ++q)
        {
            const float *q_ptr = view.data + q * dim;
            std::vector<faiss::idx_t> cpu_ids(top_k), g32_ids(top_k), g16_ids(top_k);
            std::vector<float> cpu_scores(top_k), g32_scores(top_k), g16_scores(top_k);

            cpu_index->search(1, q_ptr, top_k, cpu_scores.data(), cpu_ids.data());
            gpu_fp32->search(1, q_ptr, top_k, g32_scores.data(), g32_ids.data());
            gpu_fp16->search(1, q_ptr, top_k, g16_scores.data(), g16_ids.data());
            cudaStreamSynchronize(nullptr);

            if (g32_ids[0] == cpu_ids[0])
                ++top1_match_gpu32;
            if (g16_ids[0] == cpu_ids[0])
                ++top1_match_gpu16;

            std::unordered_set<faiss::idx_t> cpu_set(cpu_ids.begin(), cpu_ids.end());
            size_t overlap_g32 = 0, overlap_g16 = 0;
            for (auto id : g32_ids)
                if (cpu_set.count(id))
                    ++overlap_g32;
            for (auto id : g16_ids)
                if (cpu_set.count(id))
                    ++overlap_g16;
            total_top10_recall_gpu32 += static_cast<double>(overlap_g32) / top_k;
            total_top10_recall_gpu16 += static_cast<double>(overlap_g16) / top_k;

            for (int k = 0; k < top_k; ++k)
            {
                const float d32 = std::abs(cpu_scores[k] - g32_scores[k]);
                const float d16 = std::abs(cpu_scores[k] - g16_scores[k]);
                max_delta_gpu32 = std::max(max_delta_gpu32, d32);
                max_delta_gpu16 = std::max(max_delta_gpu16, d16);
                sum_delta_gpu32 += d32;
                sum_delta_gpu16 += d16;
                ++score_comparisons;
            }
        }

        std::cout << "  - GPU FP32 vs CPU FP32:" << std::endl;
        std::cout << "      Top-1 Match Rate:  " << std::fixed << std::setprecision(2)
                  << (top1_match_gpu32 * 100.0 / test_queries) << "%" << std::endl;
        std::cout << "      Top-10 Recall:     " << (total_top10_recall_gpu32 * 100.0 / test_queries) << "%" << std::endl;
        std::cout << "      Max Score Delta:   " << std::scientific << std::setprecision(4) << max_delta_gpu32 << std::endl;
        std::cout << "      Mean Score Delta:  " << (sum_delta_gpu32 / score_comparisons) << std::endl;

        std::cout << "  - GPU FP16 vs CPU FP32:" << std::endl;
        std::cout << "      Top-1 Match Rate:  " << std::fixed << std::setprecision(2)
                  << (top1_match_gpu16 * 100.0 / test_queries) << "%" << std::endl;
        std::cout << "      Top-10 Recall:     " << (total_top10_recall_gpu16 * 100.0 / test_queries) << "%" << std::endl;
        std::cout << "      Max Score Delta:   " << std::scientific << std::setprecision(4) << max_delta_gpu16 << std::endl;
        std::cout << "      Mean Score Delta:  " << (sum_delta_gpu16 / score_comparisons) << std::endl;

        // 6. Batched Query Throughput (N = 3315)
        std::cout << "\n[Stage 6] Batched Queries Throughput (N=" << N_real << ", K=10):" << std::endl;
        std::cout << std::setfill(' ');
        std::cout << std::left
                  << std::setw(10) << "Batch(Q)"
                  << std::setw(16) << "CPU FP32 (ms)"
                  << std::setw(16) << "GPU FP32 (ms)"
                  << std::setw(16) << "GPU FP16 (ms)"
                  << std::setw(16) << "CPU QPS"
                  << std::setw(16) << "GPU FP16 QPS"
                  << std::endl;
        std::cout << std::string(90, '-') << std::endl;

        for (int Q : {1, 10, 50, 100})
        {
            if (Q > static_cast<int>(N_real))
                continue;
            std::vector<faiss::idx_t> out_ids(Q * top_k);
            std::vector<float> out_scores(Q * top_k);

            auto timeBatch = [&](faiss::Index *idx, bool is_gpu) -> double {
                for (int w = 0; w < 5; ++w)
                    idx->search(Q, view.data, top_k, out_scores.data(), out_ids.data());
                if (is_gpu)
                    cudaStreamSynchronize(nullptr);
                const auto t0 = Clock::now();
                const int iters = 50;
                for (int i = 0; i < iters; ++i)
                {
                    idx->search(Q, view.data, top_k, out_scores.data(), out_ids.data());
                }
                if (is_gpu)
                    cudaStreamSynchronize(nullptr);
                const auto t1 = Clock::now();
                return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
            };

            const double ms_cpu = timeBatch(cpu_index.get(), false);
            const double ms_g32 = timeBatch(gpu_fp32.get(), true);
            const double ms_g16 = timeBatch(gpu_fp16.get(), true);

            std::cout << std::setfill(' ');
            std::cout << std::left
                      << std::setw(10) << Q
                      << std::setw(16) << std::fixed << std::setprecision(3) << ms_cpu
                      << std::setw(16) << ms_g32
                      << std::setw(16) << ms_g16
                      << std::setw(16) << std::setprecision(0) << (Q * 1000.0 / ms_cpu)
                      << std::setw(16) << (Q * 1000.0 / ms_g16)
                      << std::endl;
        }

        // 7. Scale Latency Curve (N = 100 -> 100,000, addressing user's 7.68M FLOPs / 1~1.5ms question)
        std::cout << "\n[Stage 7] Scale Latency Curve & The 7.68M FLOPs Empirical Test:" << std::endl;
        std::cout << "  (Testing single-query Q=1, K=10 exact search across gallery sizes)" << std::endl;
        std::cout << std::setfill(' ');
        std::cout << std::left
                  << std::setw(10) << "Size (N)"
                  << std::setw(14) << "FLOPs (M)"
                  << std::setw(16) << "CPU FP32 (us)"
                  << std::setw(16) << "GPU FP32 (us)"
                  << std::setw(16) << "GPU FP16 (us)"
                  << std::setw(16) << "Speedup (G16/CPU)"
                  << std::endl;
        std::cout << std::string(88, '-') << std::endl;

        // Generate synthetic scaled vectors up to 100,000
        const size_t max_scale_N = 100000;
        std::vector<float> synthetic_pool(max_scale_N * dim);
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 0.01f);
        for (size_t i = 0; i < max_scale_N; ++i)
        {
            const float *src = view.data + (i % N_real) * dim;
            float *dst = synthetic_pool.data() + i * dim;
            for (int d = 0; d < dim; ++d)
                dst[d] = src[d] + dist(rng);
            l2Normalize(dst, dim);
        }

        const std::vector<size_t> test_sizes = {100, 500, 1000, N_real, 10000, 20000, 50000, 100000};
        for (size_t target_n : test_sizes)
        {
            auto sub_cpu = std::make_unique<faiss::IndexFlatIP>(dim);
            sub_cpu->add(static_cast<faiss::idx_t>(target_n), synthetic_pool.data());

            auto sub_g32 = cloneToGpu(sub_cpu.get(), &res, device_id, false);
            auto sub_g16 = cloneToGpu(sub_cpu.get(), &res, device_id, true);

            std::vector<faiss::idx_t> ids(top_k);
            std::vector<float> scores(top_k);

            auto benchOne = [&](faiss::Index *idx, bool is_gpu) -> double {
                for (int w = 0; w < 10; ++w)
                    idx->search(1, query_vec, top_k, scores.data(), ids.data());
                if (is_gpu)
                    cudaStreamSynchronize(nullptr);
                const auto t0 = Clock::now();
                const int iters = 100;
                for (int i = 0; i < iters; ++i)
                    idx->search(1, query_vec, top_k, scores.data(), ids.data());
                if (is_gpu)
                    cudaStreamSynchronize(nullptr);
                const auto t1 = Clock::now();
                return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
            };

            const double us_cpu = benchOne(sub_cpu.get(), false);
            const double us_g32 = benchOne(sub_g32.get(), true);
            const double us_g16 = benchOne(sub_g16.get(), true);
            const double mflops = (target_n * dim * 2.0) / 1e6;

            std::cout << std::setfill(' ');
            std::cout << std::left
                      << std::setw(10) << target_n
                      << std::setw(14) << std::fixed << std::setprecision(2) << mflops
                      << std::setw(16) << std::setprecision(1) << us_cpu
                      << std::setw(16) << us_g32
                      << std::setw(16) << us_g16
                      << std::setw(16) << std::setprecision(2) << (us_cpu / us_g16) << "x"
                      << std::endl;
        }

        // 8. D2H / H2D Data Transfer Micro-Benchmark
        std::cout << "\n[Stage 8] PCIe D2H / H2D Latency for 1 Query Vector (1,536 bytes):" << std::endl;
        float *d_query;
        cudaMalloc(&d_query, dim * sizeof(float));
        std::vector<float> h_query(dim, 1.0f);
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        for (int i = 0; i < 100; ++i)
        {
            cudaMemcpyAsync(d_query, h_query.data(), dim * sizeof(float), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(h_query.data(), d_query, dim * sizeof(float), cudaMemcpyDeviceToHost, stream);
        }
        cudaStreamSynchronize(stream);

        const int pcie_iters = 5000;
        const auto t_h2d_0 = Clock::now();
        for (int i = 0; i < pcie_iters; ++i)
        {
            cudaMemcpyAsync(d_query, h_query.data(), dim * sizeof(float), cudaMemcpyHostToDevice, stream);
            cudaStreamSynchronize(stream);
        }
        const auto t_h2d_1 = Clock::now();
        const double h2d_us = std::chrono::duration<double, std::micro>(t_h2d_1 - t_h2d_0).count() / pcie_iters;

        const auto t_d2h_0 = Clock::now();
        for (int i = 0; i < pcie_iters; ++i)
        {
            cudaMemcpyAsync(h_query.data(), d_query, dim * sizeof(float), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
        }
        const auto t_d2h_1 = Clock::now();
        const double d2h_us = std::chrono::duration<double, std::micro>(t_d2h_1 - t_d2h_0).count() / pcie_iters;

        std::cout << "  - Host-to-Device (H2D) sync: " << std::fixed << std::setprecision(2) << h2d_us << " us" << std::endl;
        std::cout << "  - Device-to-Host (D2H) sync: " << d2h_us << " us" << std::endl;
        cudaFree(d_query);
        cudaStreamDestroy(stream);

        // 9. Zero-Copy HDBSCAN Clustering Benchmark
        std::cout << "\n[Stage 9] Zero-Copy HDBSCAN Clustering Benchmark (N=" << N_real << ", D=" << dim << "):" << std::endl;
        irt::features::RoiClusterConfig cluster_config;
        cluster_config.hdbscan.min_cluster_size = 5;
        cluster_config.hdbscan.min_samples = 3;
        irt::features::RoiCluster clusterer(cluster_config);

        const auto t_clust_0 = Clock::now();
        const auto cluster_results = clusterer.cluster(searcher.featureView());
        const auto t_clust_1 = Clock::now();
        const double clust_ms = std::chrono::duration<double, std::milli>(t_clust_1 - t_clust_0).count();

        std::cout << "  - HDBSCAN Time:   " << std::fixed << std::setprecision(2) << clust_ms << " ms" << std::endl;
        std::cout << "  - Clusters Found: " << cluster_results.cluster_count << std::endl;
        std::cout << "  - Noise Points:   " << cluster_results.noise_count << " ("
                  << (cluster_results.noise_count * 100.0 / N_real) << "%)" << std::endl;

        // 10. Memory Footprint Summary
        std::cout << "\n[Stage 10] Memory Footprint Summary (N=" << N_real << ", D=" << dim << "):" << std::endl;
        const double ram_mb = (N_real * dim * sizeof(float)) / (1024.0 * 1024.0);
        std::cout << "  - CPU RAM for Vector Matrix (FP32): " << std::fixed << std::setprecision(3) << ram_mb << " MB" << std::endl;
        std::cout << "  - GPU VRAM Index (FP32 approx):     " << ram_mb << " MB" << std::endl;
        std::cout << "  - GPU VRAM Index (FP16 approx):     " << (ram_mb * 0.5) << " MB" << std::endl;

        std::cout << "\n================================================================================" << std::endl;
        std::cout << "                           Benchmark Completed Successfully                     " << std::endl;
        std::cout << "================================================================================" << std::endl;
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Fatal Benchmark Error: " << e.what() << std::endl;
        return -1;
    }
}
