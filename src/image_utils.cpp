#include "cat_meme/image_utils.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>

namespace cat_meme {

cv::Mat normalizeBgr8(const cv::Mat& image) {
    if (image.empty()) {
        return {};
    }

    cv::Mat bgr;
    switch (image.channels()) {
        case 1:
            cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
            break;
        case 3:
            bgr = image;
            break;
        case 4:
            cv::cvtColor(image, bgr, cv::COLOR_BGRA2BGR);
            break;
        default:
            return {};
    }

    if (bgr.depth() == CV_8U) {
        return bgr;
    }

    cv::Mat converted;
    bgr.convertTo(converted, CV_8U);
    return converted;
}

cv::Rect centeredSubjectRect(const cv::Size& bounds, double sizeFraction) {
    const double safeFraction = std::clamp(sizeFraction, 0.1, 1.0);
    const int side = std::max(1, static_cast<int>(
        std::lround(std::min(bounds.width, bounds.height) * safeFraction)));
    return {(bounds.width - side) / 2, (bounds.height - side) / 2, side, side};
}

}  // namespace cat_meme
