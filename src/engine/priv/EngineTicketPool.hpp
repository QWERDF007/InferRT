#pragma once

#include "EngineTypes.hpp"

#include <inferrt/engine/Export.h>

#include <atomic>
#include <cstddef>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <vector>

namespace irt::engine::priv {

/** Owns all pinned host tickets used by one running engine. */
class INFERRT_ENGINE_API TicketPool final
{
public:
    struct Snapshot
    {
        size_t input_count{0};
        size_t output_count{0};
        size_t input_in_use{0};
        size_t output_in_use{0};
        size_t input_high_watermark{0};
        size_t output_high_watermark{0};
        size_t capacity_bytes{0};
        bool   aborting{false};
    };

    TicketPool(const EngineConfig &config, const std::shared_ptr<const PipelinePlan> &pipeline,
               const std::vector<size_t> &output_capacity_bytes);
    ~TicketPool() noexcept;

    TicketPool(const TicketPool &)            = delete;
    TicketPool &operator=(const TicketPool &) = delete;

    [[nodiscard]] InputTicket  *acquireInput();
    [[nodiscard]] OutputTicket *acquireOutput();
    void                        releaseInput(InputTicket *ticket);
    void                        releaseOutput(OutputTicket *ticket);

    void                  abort() noexcept;
    [[nodiscard]] Snapshot snapshot() const;
    [[nodiscard]] size_t   capacityBytes() const noexcept { return capacity_bytes_; }

private:
    void initialize(const EngineConfig &config, const std::shared_ptr<const PipelinePlan> &pipeline,
                    const std::vector<size_t> &output_capacity_bytes);

    std::vector<std::unique_ptr<InputTicket>>  input_tickets_;
    std::deque<InputTicket *>                  free_input_tickets_;
    mutable std::mutex                        input_mutex_;
    std::condition_variable                   input_condition_;
    size_t                                     input_in_use_{0};
    size_t                                     input_high_watermark_{0};

    std::vector<std::unique_ptr<OutputTicket>> output_tickets_;
    std::deque<OutputTicket *>                 free_output_tickets_;
    mutable std::mutex                        output_mutex_;
    std::condition_variable                   output_condition_;
    size_t                                     output_in_use_{0};
    size_t                                     output_high_watermark_{0};

    size_t                  capacity_bytes_{0};
    std::atomic_bool        aborting_{false};
};

} // namespace irt::engine::priv
