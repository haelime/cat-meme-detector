#pragma once

#include <opencv2/core.hpp>
#include <opencv2/features.hpp>

namespace cat_meme {

struct VisualFeature {
    cv::Mat colorHistogram;
    cv::Mat gradientDescriptor;
    cv::Mat orbDescriptors;
};

struct SimilarityBreakdown {
    double total = 0.0;
    double color = 0.0;
    double shape = 0.0;
    double local = 0.0;
};

class FeatureExtractor {
public:
    FeatureExtractor();

    [[nodiscard]] VisualFeature extract(const cv::Mat& bgrImage) const;
    [[nodiscard]] SimilarityBreakdown compare(const VisualFeature& first,
                                              const VisualFeature& second) const;

private:
    cv::Ptr<cv::ORB> orb_;
};

}  // namespace cat_meme
