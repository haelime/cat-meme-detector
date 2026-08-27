#pragma once

#include <filesystem>
#include <optional>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/xobjdetect.hpp>

namespace cat_meme {

class FaceDetector {
public:
    explicit FaceDetector(const std::filesystem::path& cascadePath);

    [[nodiscard]] std::vector<cv::Rect> detect(const cv::Mat& bgrImage);
    [[nodiscard]] std::optional<cv::Rect> largestFace(const cv::Mat& bgrImage);

private:
    cv::CascadeClassifier cascade_;
};

[[nodiscard]] cv::Rect paddedRect(const cv::Rect& rect, const cv::Size& bounds,
                                  double paddingFraction = 0.18);

}  // namespace cat_meme
