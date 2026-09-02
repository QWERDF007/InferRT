#include "../../samples/engine/engine_basic/EngineExample.hpp"

#include <gtest/gtest.h>

#include <filesystem>

namespace {

TEST(EngineSampleConfigTest, DefaultsToPortableSingleBatch)
{
    const auto config = irt::sample::engine::makeConfig("resnet18", std::filesystem::path("resnet18.engine"), 0);

    EXPECT_EQ(config.min_batch_size, 1);
    EXPECT_EQ(config.opt_batch_size, 1);
    EXPECT_EQ(config.max_batch_size, 1);
}

TEST(EngineSampleConfigTest, PreservesExplicitDynamicBatchRange)
{
    const auto config = irt::sample::engine::makeConfig("resnet18", std::filesystem::path("resnet18.engine"), 0,
                                                        0, 1, 4, 8);

    EXPECT_EQ(config.min_batch_size, 1);
    EXPECT_EQ(config.opt_batch_size, 4);
    EXPECT_EQ(config.max_batch_size, 8);
}

} // namespace
