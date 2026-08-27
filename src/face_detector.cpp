#include "cat_meme/face_detector.hpp"

#include <algorithm>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace cat_meme {

FaceDetector::FaceDetector(const std::filesystem::path& cascadePath) {
    if (!cascade_.load(cascadePath.string())) {
        throw std::runtime_error("Could not load human-face cascade: " +
                                 cascadePath.string());
    }
}

std::vector<cv::Rect> FaceDetector::detect(const cv::Mat& bgrImage) {
    if (bgrImage.empty()) {
        return {};
    }

    cv::Mat gray;
    if (bgrImage.channels() == 1) {
        gray = bgrImage;
    } else {
        cv::cvtColor(bgrImage, gray, cv::COLOR_BGR2GRAY);
    }

    cv::Mat enhanced;
    cv::createCLAHE(2.0, cv::Size(8, 8))->apply(gray, enhanced);

    std::vector<cv::Rect> faces;
    const int minSide = std::max(32, std::min(enhanced.cols, enhanced.rows) / 12);
    cascade_.detectMultiScale(enhanced, faces, 1.06, 3, 0,
                              cv::Size(minSide, minSide), cv::Size());
    return faces;
}

std::optional<cv::Rect> FaceDetector::largestFace(const cv::Mat& bgrImage) {
    auto faces = detect(bgrImage);
    if (faces.empty()) {
        return std::nullopt;
    }

    return *std::max_element(faces.begin(), faces.end(),
                             [](const cv::Rect& a, const cv::Rect& b) {
                                 return a.area() < b.area();
                             });
}

cv::Rect paddedRect(const cv::Rect& rect, const cv::Size& bounds,
                    double paddingFraction) {
    const int padX = static_cast<int>(rect.width * paddingFraction);
    const int padY = static_cast<int>(rect.height * paddingFraction);
    const int x = std::max(0, rect.x - padX);
    const int y = std::max(0, rect.y - padY);
    const int right = std::min(bounds.width, rect.x + rect.width + padX);
    const int bottom = std::min(bounds.height, rect.y + rect.height + padY);
    return {x, y, std::max(1, right - x), std::max(1, bottom - y)};
}

}  // namespace cat_meme
