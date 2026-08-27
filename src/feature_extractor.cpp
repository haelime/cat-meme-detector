#include "cat_meme/feature_extractor.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace cat_meme {
namespace {

cv::Mat normalizedSquare(const cv::Mat& image) {
    cv::Mat bgr;
    if (image.channels() == 4) {
        cv::cvtColor(image, bgr, cv::COLOR_BGRA2BGR);
    } else if (image.channels() == 1) {
        cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
    } else {
        bgr = image;
    }
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(256, 256), 0.0, 0.0, cv::INTER_AREA);
    return resized;
}

cv::Mat colorHistogram(const cv::Mat& bgr) {
    cv::Mat hsv;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
    const int channels[] = {0, 1};
    const int bins[] = {24, 16};
    const float hueRange[] = {0.0F, 180.0F};
    const float saturationRange[] = {0.0F, 256.0F};
    const float* ranges[] = {hueRange, saturationRange};
    cv::Mat histogram;
    cv::calcHist(&hsv, 1, channels, cv::Mat(), histogram, 2, bins, ranges, true, false);
    cv::normalize(histogram, histogram, 1.0, 0.0, cv::NORM_L1);
    return histogram;
}

cv::Mat gradientDescriptor(const cv::Mat& bgr) {
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0.0);

    cv::Mat dx;
    cv::Mat dy;
    cv::Sobel(gray, dx, CV_32F, 1, 0, 3);
    cv::Sobel(gray, dy, CV_32F, 0, 1, 3);

    constexpr int grid = 4;
    constexpr int bins = 12;
    cv::Mat descriptor = cv::Mat::zeros(1, grid * grid * bins, CV_32F);
    const int cellWidth = gray.cols / grid;
    const int cellHeight = gray.rows / grid;

    for (int y = 0; y < gray.rows; ++y) {
        const float* rowX = dx.ptr<float>(y);
        const float* rowY = dy.ptr<float>(y);
        for (int x = 0; x < gray.cols; ++x) {
            const float magnitude = std::hypot(rowX[x], rowY[x]);
            float angle = std::atan2(rowY[x], rowX[x]);
            if (angle < 0.0F) {
                angle += static_cast<float>(CV_PI);
            }
            if (angle >= static_cast<float>(CV_PI)) {
                angle -= static_cast<float>(CV_PI);
            }
            const int cellX = std::min(grid - 1, x / cellWidth);
            const int cellY = std::min(grid - 1, y / cellHeight);
            const int bin = std::min(bins - 1,
                                     static_cast<int>(angle / static_cast<float>(CV_PI) * bins));
            descriptor.at<float>(0, (cellY * grid + cellX) * bins + bin) += magnitude;
        }
    }
    cv::normalize(descriptor, descriptor, 1.0, 0.0, cv::NORM_L2);
    return descriptor;
}

double clampUnit(double value) {
    return std::clamp(value, 0.0, 1.0);
}

}  // namespace

FeatureExtractor::FeatureExtractor()
    : orb_(cv::ORB::create(900, 1.2F, 8, 20, 0, 2, cv::ORB::HARRIS_SCORE, 20, 7)) {}

VisualFeature FeatureExtractor::extract(const cv::Mat& bgrImage) const {
    if (bgrImage.empty()) {
        return {};
    }

    const cv::Mat prepared = normalizedSquare(bgrImage);
    cv::Mat gray;
    cv::cvtColor(prepared, gray, cv::COLOR_BGR2GRAY);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    orb_->detectAndCompute(gray, cv::Mat(), keypoints, descriptors);

    return {colorHistogram(prepared), gradientDescriptor(prepared), descriptors};
}

SimilarityBreakdown FeatureExtractor::compare(const VisualFeature& first,
                                               const VisualFeature& second) const {
    SimilarityBreakdown result;
    if (first.colorHistogram.empty() || second.colorHistogram.empty()) {
        return result;
    }

    result.color = clampUnit(1.0 - cv::compareHist(first.colorHistogram,
                                                   second.colorHistogram,
                                                   cv::HISTCMP_BHATTACHARYYA));

    if (!first.gradientDescriptor.empty() && !second.gradientDescriptor.empty()) {
        result.shape = clampUnit(first.gradientDescriptor.dot(second.gradientDescriptor));
    }

    if (!first.orbDescriptors.empty() && !second.orbDescriptors.empty()) {
        cv::BFMatcher matcher(cv::NORM_HAMMING);
        std::vector<std::vector<cv::DMatch>> matches;
        matcher.knnMatch(first.orbDescriptors, second.orbDescriptors, matches, 2);
        int good = 0;
        for (const auto& pair : matches) {
            if (pair.size() == 2 && pair[0].distance < 0.75F * pair[1].distance) {
                ++good;
            }
        }
        const int denominator = std::max(1, std::min(first.orbDescriptors.rows,
                                                     second.orbDescriptors.rows));
        result.local = clampUnit(static_cast<double>(good) / denominator * 3.0);
    }

    result.total = 100.0 * (0.45 * result.color + 0.35 * result.shape + 0.20 * result.local);
    return result;
}

}  // namespace cat_meme

