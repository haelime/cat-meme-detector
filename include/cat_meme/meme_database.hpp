#pragma once

#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "cat_meme/feature_extractor.hpp"
#include "cat_meme/semantic_analysis.hpp"

namespace cat_meme {

struct Meme {
    std::filesystem::path path;
    std::string displayName;
    cv::Mat preview;
    VisualFeature feature;
    SemanticProfile semantic;
    bool animated = false;
};

struct Match {
    const Meme* meme = nullptr;
    SimilarityBreakdown similarity;
    double visualScore = 0.0;
    double expressionScore = 0.0;
    double gestureScore = 0.0;
};

[[nodiscard]] double weightedMatchScore(double visual, double expression,
                                        double gesture,
                                        bool handDetected) noexcept;

class MemeDatabase {
public:
    using ProgressCallback = std::function<void(
        std::size_t, std::size_t, const std::filesystem::path&)>;
    using SemanticAnalysisCallback = std::function<SceneAnalysis(const cv::Mat&)>;

    explicit MemeDatabase(FeatureExtractor& extractor);

    std::size_t load(const std::filesystem::path& directory,
                     const ProgressCallback& progress = {},
                     const SemanticAnalysisCallback& analyzeSemantic = {});
    [[nodiscard]] Match bestMatch(const cv::Mat& subject,
                                  const SceneAnalysis* analysis = nullptr) const;
    [[nodiscard]] const std::vector<Meme>& memes() const noexcept;

private:
    FeatureExtractor& extractor_;
    std::vector<Meme> memes_;
};

[[nodiscard]] bool isSupportedImage(const std::filesystem::path& path);
[[nodiscard]] cv::Mat loadFirstFrame(const std::filesystem::path& path);
[[nodiscard]] std::vector<std::pair<std::string, SemanticProfile>> loadMemeLabels(
    const std::filesystem::path& path);

}  // namespace cat_meme
