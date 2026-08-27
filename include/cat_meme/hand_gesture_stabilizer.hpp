#pragma once

#include <cstddef>

#include "cat_meme/semantic_analysis.hpp"

namespace cat_meme {

class HandGestureStabilizer {
public:
    explicit HandGestureStabilizer(std::size_t confirmations = 4,
                                   std::size_t releases = 2) noexcept;

    [[nodiscard]] Gesture update(Gesture observed) noexcept;
    void reset() noexcept;

private:
    std::size_t confirmations_;
    std::size_t releases_;
    Gesture candidate_ = Gesture::None;
    Gesture confirmed_ = Gesture::None;
    std::size_t candidateCount_ = 0;
    std::size_t missingCount_ = 0;
};

}  // namespace cat_meme
