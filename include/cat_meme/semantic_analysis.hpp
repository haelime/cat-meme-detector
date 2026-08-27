#pragma once

#include <array>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

namespace cat_meme {

enum class Expression {
    Angry,
    Disgust,
    Fearful,
    Happy,
    Neutral,
    Sad,
    Surprised,
    Unknown
};

enum class Gesture {
    None,
    Fist,
    OpenPalm,
    Peace,
    Pointing,
    ThumbsUp,
    Other
};

constexpr std::size_t kExpressionCount = 7;

struct SemanticProfile {
    Expression expression = Expression::Unknown;
    Gesture gesture = Gesture::None;
};

struct SceneAnalysis {
    std::array<float, kExpressionCount> expressionProbabilities{};
    Expression expression = Expression::Unknown;
    float expressionConfidence = 0.0F;
    Gesture gesture = Gesture::None;
    float handConfidence = 0.0F;
    cv::Rect handBox;
    std::vector<cv::Point3f> handLandmarks;
};

[[nodiscard]] const char* toString(Expression expression) noexcept;
[[nodiscard]] const char* toString(Gesture gesture) noexcept;
[[nodiscard]] Expression parseExpression(const std::string& value);
[[nodiscard]] Gesture parseGesture(const std::string& value);
[[nodiscard]] Gesture classifyGesture(const std::vector<cv::Point3f>& landmarks);
[[nodiscard]] double expressionScore(const SceneAnalysis& analysis,
                                     Expression target) noexcept;
[[nodiscard]] double gestureScore(const SceneAnalysis& analysis,
                                  Gesture target) noexcept;

class DnnAnalyzer {
public:
    using LoadProgressCallback = std::function<void(
        std::size_t current, std::size_t total,
        const std::filesystem::path& model)>;

    DnnAnalyzer(const std::filesystem::path& expressionModel,
                const std::filesystem::path& palmModel,
                const std::filesystem::path& handPoseModel,
                bool preferCuda = true,
                const LoadProgressCallback& progress = {});

    [[nodiscard]] SceneAnalysis analyze(const cv::Mat& bgrFrame,
                                        const cv::Rect& face);
    [[nodiscard]] const std::string& backendName() const noexcept;
    [[nodiscard]] bool usingCuda() const noexcept;

private:
    struct Palm {
        cv::Rect2f box;
        std::array<cv::Point2f, 7> landmarks{};
        float confidence = 0.0F;
    };

    void configureNetworks(bool cuda);
    cv::Mat forward(cv::dnn::Net& network, const cv::Mat& input,
                    const std::string& outputName = {});
    std::vector<cv::Mat> forwardAll(cv::dnn::Net& network, const cv::Mat& input);
    void analyzeExpression(const cv::Mat& frame, const cv::Rect& face,
                           SceneAnalysis& result);
    bool detectPalm(const cv::Mat& frame, Palm& palm);
    bool estimateHand(const cv::Mat& frame, const Palm& palm,
                      SceneAnalysis& result);

    cv::dnn::Net expressionNet_;
    cv::dnn::Net palmNet_;
    cv::dnn::Net handPoseNet_;
    std::vector<cv::Point2f> palmAnchors_;
    bool cuda_ = false;
    bool fallbackReported_ = false;
    std::string backendName_ = "CPU";
};

}  // namespace cat_meme
