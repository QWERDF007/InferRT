#pragma once

/**
 * @file DinoContracts.hpp
 * @brief YAML contracts for profile, request, response, and reports.
 */

#include <inferrt/features/DinoRegionSearch.hpp>

namespace irt::features::priv {

/** @brief Search status and decision names. */
const char *dinoStatusName(DinoSearchStatus status) noexcept;
const char *dinoDecisionName(DinoSearchDecision decision) noexcept;

} // namespace irt::features::priv
