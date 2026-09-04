#include <benchmark/benchmark.h>
#include <inferrt/features/ShapeTemplateMatcher.hpp>

#include "ShapeTemplateMatcherFactory.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <stdexcept>

namespace {

cv::Mat MakeTemplate()
{
    cv::Mat image(128, 128, CV_8UC1, cv::Scalar(0));
    cv::rectangle(image, {18, 24}, {102, 92}, cv::Scalar(255), 3);
    cv::line(image, {18, 24}, {102, 92}, cv::Scalar(180), 2);
    cv::circle(image, {82, 46}, 12, cv::Scalar(220), 2);
    return image;
}
cv::Mat MakeSource()
{
    cv::Mat image(480, 640, CV_8UC1, cv::Scalar(15));
    cv::RNG rng(0x5A17); rng.fill(image, cv::RNG::NORMAL, 15, 5);
    MakeTemplate().copyTo(image(cv::Rect(244, 156, 128, 128)));
    return image;
}
irt::features::ShapeTemplateMatcherConfig Config()
{
    irt::features::ShapeTemplateMatcherConfig c; c.num_features = 96; c.scan_step = 1; c.match_threshold = 80.0f; c.nms_threshold = 0.3f; return c;
}
bool SupportsAvx512()
{
    return irt::features::priv::shapeTemplateMatcherImplementationSupported(
        irt::features::priv::ShapeTemplateMatcherImplementation::Avx512);
}
void VerifyParity()
{
    static const bool verified = []
    {
        const cv::Mat templ = MakeTemplate();
        const cv::Mat source = MakeSource();
        auto avx2 = irt::features::priv::createShapeTemplateMatcherForImplementation(
            irt::features::priv::ShapeTemplateMatcherImplementation::Avx2, Config());
        auto scalar = irt::features::priv::createShapeTemplateMatcherForImplementation(
            irt::features::priv::ShapeTemplateMatcherImplementation::Scalar, Config());
        avx2->addTemplate(templ);
        scalar->addTemplate(templ);
        const auto avx2_matches = avx2->match(source);
        const auto scalar_matches = scalar->match(source);
        if (avx2_matches.size() != scalar_matches.size())
            throw std::logic_error("AVX2 and scalar shape-template matches differ");
        for (size_t i = 0; i < avx2_matches.size(); ++i)
        {
            const auto &a = avx2_matches[i];
            const auto &b = scalar_matches[i];
            if (a.x != b.x || a.y != b.y || a.width != b.width || a.height != b.height
                || a.similarity != b.similarity || a.template_id != b.template_id
                || a.angle_degrees != b.angle_degrees || a.scale != b.scale)
                throw std::logic_error("AVX2 and scalar shape-template match contents differ");
        }
        if (SupportsAvx512())
        {
            auto avx512 = irt::features::priv::createShapeTemplateMatcherForImplementation(
                irt::features::priv::ShapeTemplateMatcherImplementation::Avx512, Config());
            avx512->addTemplate(templ);
            const auto avx512_matches = avx512->match(source);
            if (avx512_matches.size() != avx2_matches.size())
                throw std::logic_error("AVX512 and AVX2 shape-template matches differ");
            for (size_t i = 0; i < avx512_matches.size(); ++i)
            {
                const auto &a = avx512_matches[i];
                const auto &b = avx2_matches[i];
                if (a.x != b.x || a.y != b.y || a.width != b.width || a.height != b.height
                    || a.similarity != b.similarity || a.template_id != b.template_id
                    || a.angle_degrees != b.angle_degrees || a.scale != b.scale)
                    throw std::logic_error("AVX512 and AVX2 shape-template match contents differ");
            }
        }
        return true;
    }();
    (void)verified;
}
void BenchmarkMatch(benchmark::State &state, irt::features::priv::ShapeTemplateMatcherImplementation implementation,
                    int template_count = 1)
{
    VerifyParity();
    const cv::Mat templ = MakeTemplate();
    const cv::Mat source = MakeSource();
    auto matcher = irt::features::priv::createShapeTemplateMatcherForImplementation(implementation, Config());
    for (int i = 0; i < template_count; ++i)
        matcher->addTemplate(templ);
    for (auto _ : state)
    {
        auto matches = matcher->match(source);
        benchmark::DoNotOptimize(matches.data());
        benchmark::ClobberMemory();
    }
    state.SetLabel("640x480, 96 features, scan_step=1, templates=" + std::to_string(template_count));
}
void AVX2(benchmark::State &state)
{
    BenchmarkMatch(state, irt::features::priv::ShapeTemplateMatcherImplementation::Avx2);
}
void AVX2MultiTemplate(benchmark::State &state)
{
    BenchmarkMatch(state, irt::features::priv::ShapeTemplateMatcherImplementation::Avx2, 4);
}
void AVX512(benchmark::State &state)
{
    if (!SupportsAvx512()) { state.SkipWithError("AVX512F/BW is not available on this CPU"); return; }
    BenchmarkMatch(state, irt::features::priv::ShapeTemplateMatcherImplementation::Avx512);
}
void AVX512MultiTemplate(benchmark::State &state)
{
    if (!SupportsAvx512()) { state.SkipWithError("AVX512F/BW is not available on this CPU"); return; }
    BenchmarkMatch(state, irt::features::priv::ShapeTemplateMatcherImplementation::Avx512, 4);
}
void Scalar(benchmark::State &state)
{
    BenchmarkMatch(state, irt::features::priv::ShapeTemplateMatcherImplementation::Scalar);
}
void ScalarMultiTemplate(benchmark::State &state)
{
    BenchmarkMatch(state, irt::features::priv::ShapeTemplateMatcherImplementation::Scalar, 4);
}
BENCHMARK(AVX2)->Unit(benchmark::kMillisecond);
BENCHMARK(AVX2MultiTemplate)->Unit(benchmark::kMillisecond);
BENCHMARK(AVX512)->Unit(benchmark::kMillisecond);
BENCHMARK(AVX512MultiTemplate)->Unit(benchmark::kMillisecond);
BENCHMARK(Scalar)->Unit(benchmark::kMillisecond);
BENCHMARK(ScalarMultiTemplate)->Unit(benchmark::kMillisecond);
} // namespace
BENCHMARK_MAIN();
