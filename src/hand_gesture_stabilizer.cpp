#include "cat_meme/hand_gesture_stabilizer.hpp"

#include <algorithm>

namespace cat_meme {

HandGestureStabilizer::HandGestureStabilizer(std::size_t confirmations,
                                             std::size_t releases) noexcept
    : confirmations_(std::max<std::size_t>(1, confirmations)),
      releases_(std::max<std::size_t>(1, releases)) {}

Gesture HandGestureStabilizer::update(Gesture observed) noexcept {
    // Counts are analysis observations, not camera frames; main.cpp samples every fifth frame.
    if (observed == Gesture::None) {
        candidate_ = Gesture::None;
        candidateCount_ = 0;
        // Brief detection gaps keep the last confirmed gesture until the release threshold.
        if (confirmed_ != Gesture::None) {
            ++missingCount_;
            if (missingCount_ >= releases_) {
                confirmed_ = Gesture::None;
                missingCount_ = 0;
            }
        }
        return confirmed_;
    }

    missingCount_ = 0;
    if (observed == confirmed_) {
        candidate_ = observed;
        candidateCount_ = confirmations_;
        return confirmed_;
    }

    if (observed != candidate_) {
        candidate_ = observed;
        candidateCount_ = 1;
    } else {
        ++candidateCount_;
    }
    if (candidateCount_ >= confirmations_) {
        confirmed_ = candidate_;
    }
    return confirmed_;
}

void HandGestureStabilizer::reset() noexcept {
    candidate_ = Gesture::None;
    confirmed_ = Gesture::None;
    candidateCount_ = 0;
    missingCount_ = 0;
}

}  // namespace cat_meme
