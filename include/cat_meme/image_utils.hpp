#pragma once

#include <opencv2/core.hpp>

namespace cat_meme {

// Converts decoded still/video frames to the format expected by the UI.
// Returns an empty matrix for unsupported channel layouts.
[[nodiscard]] cv::Mat normalizeBgr8(const cv::Mat& image);
[[nodiscard]] cv::Rect centeredSubjectRect(const cv::Size& bounds,
                                           double sizeFraction = 0.72);

}  // namespace cat_meme
