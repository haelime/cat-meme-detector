#include "cat_meme/semantic_analysis.hpp"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <iostream>
#include <limits>
#include <stdexcept>

#include <opencv2/imgproc.hpp>
#include <opencv2/geometry/2d.hpp>

namespace cat_meme {
namespace {

constexpr int kPalmInput = 192;
constexpr int kHandInput = 224;

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

cv::Rect clipped(const cv::Rect& rect, const cv::Size& size) {
    return rect & cv::Rect(0, 0, size.width, size.height);
}

cv::Mat nhwcFloat(const cv::Mat& rgbFloat) {
    CV_Assert(rgbFloat.type() == CV_32FC3 && rgbFloat.isContinuous());
    const int dimensions[] = {1, rgbFloat.rows, rgbFloat.cols, 3};
    return cv::Mat(4, dimensions, CV_32F,
                   const_cast<float*>(rgbFloat.ptr<float>())).clone();
}

float sigmoid(float value) {
    value = std::clamp(value, -80.0F, 80.0F);
    return 1.0F / (1.0F + std::exp(-value));
}

double distance2d(const cv::Point3f& first, const cv::Point3f& second) {
    return std::hypot(static_cast<double>(first.x - second.x),
                      static_cast<double>(first.y - second.y));
}

}  // namespace

const char* toString(Expression expression) noexcept {
    static constexpr const char* names[] = {
        "angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised"
    };
    const auto index = static_cast<std::size_t>(expression);
    return index < kExpressionCount ? names[index] : "unknown";
}

const char* toString(Gesture gesture) noexcept {
    switch (gesture) {
        case Gesture::None: return "none";
        case Gesture::Fist: return "fist";
        case Gesture::OpenPalm: return "open-palm";
        case Gesture::Peace: return "peace";
        case Gesture::Pointing: return "pointing";
        case Gesture::ThumbsUp: return "thumbs-up";
        case Gesture::Other: return "other";
    }
    return "other";
}

Expression parseExpression(const std::string& value) {
    const std::string normalized = lower(value);
    for (std::size_t index = 0; index < kExpressionCount; ++index) {
        const auto expression = static_cast<Expression>(index);
        if (normalized == toString(expression)) {
            return expression;
        }
    }
    if (normalized == "unknown" || normalized.empty()) {
        return Expression::Unknown;
    }
    throw std::runtime_error("Unknown expression label: " + value);
}

Gesture parseGesture(const std::string& value) {
    const std::string normalized = lower(value);
    if (normalized.empty() || normalized == "none") return Gesture::None;
    if (normalized == "fist") return Gesture::Fist;
    if (normalized == "open-palm" || normalized == "open_palm") return Gesture::OpenPalm;
    if (normalized == "peace") return Gesture::Peace;
    if (normalized == "pointing") return Gesture::Pointing;
    if (normalized == "thumbs-up" || normalized == "thumbs_up") return Gesture::ThumbsUp;
    if (normalized == "other") return Gesture::Other;
    throw std::runtime_error("Unknown gesture label: " + value);
}

Gesture classifyGesture(const std::vector<cv::Point3f>& landmarks) {
    if (landmarks.size() < 21) {
        return Gesture::None;
    }
    const cv::Point3f& wrist = landmarks[0];
    const bool thumb = distance2d(landmarks[4], wrist) >
                       distance2d(landmarks[3], wrist) * 1.08;
    const bool index = distance2d(landmarks[8], wrist) >
                       distance2d(landmarks[6], wrist) * 1.12;
    const bool middle = distance2d(landmarks[12], wrist) >
                        distance2d(landmarks[10], wrist) * 1.12;
    const bool ring = distance2d(landmarks[16], wrist) >
                      distance2d(landmarks[14], wrist) * 1.12;
    const bool pinky = distance2d(landmarks[20], wrist) >
                       distance2d(landmarks[18], wrist) * 1.12;

    if (!index && !middle && !ring && !pinky) {
        return thumb ? Gesture::ThumbsUp : Gesture::Fist;
    }
    if (thumb && index && middle && ring && pinky) {
        return Gesture::OpenPalm;
    }
    if (index && middle && !ring && !pinky) {
        return Gesture::Peace;
    }
    if (index && !middle && !ring && !pinky) {
        return Gesture::Pointing;
    }
    return Gesture::Other;
}

double expressionScore(const SceneAnalysis& analysis, Expression target) noexcept {
    const auto index = static_cast<std::size_t>(target);
    if (analysis.expression == Expression::Unknown || index >= kExpressionCount) {
        return 0.0;
    }
    return 100.0 * std::clamp(static_cast<double>(analysis.expressionProbabilities[index]),
                              0.0, 1.0);
}

double gestureScore(const SceneAnalysis& analysis, Gesture target) noexcept {
    if (analysis.gesture == Gesture::None) {
        return 0.0;
    }
    if (analysis.gesture == target) {
        return 100.0;
    }
    if (analysis.gesture == Gesture::Other || target == Gesture::Other) {
        return 35.0;
    }
    return 0.0;
}

DnnAnalyzer::DnnAnalyzer(const std::filesystem::path& expressionModel,
                         const std::filesystem::path& palmModel,
                         const std::filesystem::path& handPoseModel,
                         bool preferCuda,
                         const LoadProgressCallback& progress) {
    for (const auto& path : {expressionModel, palmModel, handPoseModel}) {
        if (!std::filesystem::is_regular_file(path)) {
            throw std::runtime_error("DNN model is missing: " + path.string());
        }
    }
    if (progress) progress(0, 3, {});
    // OpenCV 5's new graph engine is CPU-only. The classic engine is required
    // for setPreferableBackend(DNN_BACKEND_CUDA) to take effect.
    expressionNet_ = cv::dnn::readNetFromONNX(
        expressionModel.string(), cv::dnn::ENGINE_CLASSIC);
    if (progress) progress(1, 3, expressionModel);
    palmNet_ = cv::dnn::readNetFromONNX(
        palmModel.string(), cv::dnn::ENGINE_CLASSIC);
    if (progress) progress(2, 3, palmModel);
    handPoseNet_ = cv::dnn::readNetFromONNX(
        handPoseModel.string(), cv::dnn::ENGINE_CLASSIC);
    if (progress) progress(3, 3, handPoseModel);

    palmAnchors_.reserve(2016);
    for (int y = 0; y < 24; ++y) {
        for (int x = 0; x < 24; ++x) {
            for (int repeat = 0; repeat < 2; ++repeat) {
                palmAnchors_.emplace_back((x + 0.5F) / 24.0F, (y + 0.5F) / 24.0F);
            }
        }
    }
    for (int y = 0; y < 12; ++y) {
        for (int x = 0; x < 12; ++x) {
            for (int repeat = 0; repeat < 6; ++repeat) {
                palmAnchors_.emplace_back((x + 0.5F) / 12.0F, (y + 0.5F) / 12.0F);
            }
        }
    }

    bool cudaAvailable = false;
    if (preferCuda) {
        try {
            cudaAvailable = !cv::dnn::getAvailableTargets(cv::dnn::DNN_BACKEND_CUDA).empty();
        } catch (const cv::Exception&) {
            cudaAvailable = false;
        }
    }
    configureNetworks(cudaAvailable);
}

void DnnAnalyzer::configureNetworks(bool cuda) {
    cuda_ = cuda;
    backendName_ = cuda ? "CUDA FP16" : "CPU";
    const int backend = cuda ? cv::dnn::DNN_BACKEND_CUDA : cv::dnn::DNN_BACKEND_OPENCV;
    const int target = cuda ? cv::dnn::DNN_TARGET_CUDA_FP16 : cv::dnn::DNN_TARGET_CPU;
    for (cv::dnn::Net* network : {&expressionNet_, &palmNet_, &handPoseNet_}) {
        network->setPreferableBackend(backend);
        network->setPreferableTarget(target);
    }
}

cv::Mat DnnAnalyzer::forward(cv::dnn::Net& network, const cv::Mat& input,
                             const std::string& outputName) {
    try {
        network.setInput(input);
        return outputName.empty() ? network.forward() : network.forward(outputName);
    } catch (const cv::Exception&) {
        if (!cuda_) throw;
        configureNetworks(false);
        if (!fallbackReported_) {
            std::cerr << "CUDA DNN inference failed; switched to CPU\n";
            fallbackReported_ = true;
        }
        network.setInput(input);
        return outputName.empty() ? network.forward() : network.forward(outputName);
    }
}

std::vector<cv::Mat> DnnAnalyzer::forwardAll(cv::dnn::Net& network,
                                             const cv::Mat& input) {
    std::vector<cv::Mat> outputs;
    const std::vector<cv::String> names = network.getUnconnectedOutLayersNames();
    try {
        network.setInput(input);
        network.forward(outputs, names);
    } catch (const cv::Exception&) {
        if (!cuda_) throw;
        configureNetworks(false);
        if (!fallbackReported_) {
            std::cerr << "CUDA DNN inference failed; switched to CPU\n";
            fallbackReported_ = true;
        }
        network.setInput(input);
        network.forward(outputs, names);
    }
    return outputs;
}

SceneAnalysis DnnAnalyzer::analyze(const cv::Mat& bgrFrame, const cv::Rect& face) {
    SceneAnalysis result;
    if (bgrFrame.empty()) return result;
    if (face.area() > 0) analyzeExpression(bgrFrame, face, result);
    Palm palm;
    if (detectPalm(bgrFrame, palm)) {
        estimateHand(bgrFrame, palm, result);
    }
    return result;
}

void DnnAnalyzer::analyzeExpression(const cv::Mat& frame, const cv::Rect& face,
                                    SceneAnalysis& result) {
    const cv::Rect area = clipped(face, frame.size());
    if (area.empty()) return;
    cv::Mat normalized;
    cv::resize(frame(area), normalized, cv::Size(112, 112), 0.0, 0.0, cv::INTER_AREA);
    cv::Mat input = cv::dnn::blobFromImage(normalized, 1.0 / 127.5,
                                           cv::Size(), cv::Scalar(127.5, 127.5, 127.5));
    cv::Mat output;
    try {
        output = forward(expressionNet_, input, "label");
    } catch (const cv::Exception&) {
        output = forward(expressionNet_, input);
    }
    output = output.reshape(1, 1);
    if (output.total() < kExpressionCount) return;

    float maximum = -std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < kExpressionCount; ++i) {
        maximum = std::max(maximum, output.at<float>(static_cast<int>(i)));
    }
    float sum = 0.0F;
    for (std::size_t i = 0; i < kExpressionCount; ++i) {
        result.expressionProbabilities[i] =
            std::exp(output.at<float>(static_cast<int>(i)) - maximum);
        sum += result.expressionProbabilities[i];
    }
    std::size_t best = 0;
    for (std::size_t i = 0; i < kExpressionCount; ++i) {
        result.expressionProbabilities[i] /= sum;
        if (result.expressionProbabilities[i] > result.expressionProbabilities[best]) best = i;
    }
    result.expression = static_cast<Expression>(best);
    result.expressionConfidence = result.expressionProbabilities[best];
}

bool DnnAnalyzer::detectPalm(const cv::Mat& frame, Palm& palm) {
    const float ratio = std::min(static_cast<float>(kPalmInput) / frame.rows,
                                 static_cast<float>(kPalmInput) / frame.cols);
    const cv::Size resizedSize(std::max(1, cvRound(frame.cols * ratio)),
                               std::max(1, cvRound(frame.rows * ratio)));
    cv::Mat resized;
    cv::resize(frame, resized, resizedSize, 0.0, 0.0, cv::INTER_AREA);
    const int left = (kPalmInput - resized.cols) / 2;
    const int top = (kPalmInput - resized.rows) / 2;
    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, top, kPalmInput - resized.rows - top,
                       left, kPalmInput - resized.cols - left,
                       cv::BORDER_CONSTANT, cv::Scalar());
    cv::cvtColor(padded, padded, cv::COLOR_BGR2RGB);
    padded.convertTo(padded, CV_32FC3, 1.0 / 255.0);
    std::vector<cv::Mat> outputs = forwardAll(palmNet_, nhwcFloat(padded));
    if (outputs.size() < 2) return false;

    cv::Mat boxes;
    cv::Mat scores;
    for (cv::Mat& output : outputs) {
        cv::Mat flat = output.reshape(1, static_cast<int>(palmAnchors_.size()));
        if (flat.cols >= 18) boxes = flat;
        else if (flat.cols == 1) scores = flat;
    }
    if (boxes.empty() || scores.empty()) return false;

    const float scale = static_cast<float>(std::max(frame.cols, frame.rows));
    const cv::Point2f bias(left / ratio, top / ratio);
    std::vector<cv::Rect> candidateBoxes;
    std::vector<float> candidateScores;
    std::vector<int> sourceIndices;
    for (int i = 0; i < boxes.rows; ++i) {
        const float confidence = sigmoid(scores.at<float>(i, 0));
        if (confidence < 0.25F) continue;
        const cv::Point2f center(boxes.at<float>(i, 0) / kPalmInput + palmAnchors_[i].x,
                                 boxes.at<float>(i, 1) / kPalmInput + palmAnchors_[i].y);
        const cv::Point2f size(boxes.at<float>(i, 2) / kPalmInput,
                               boxes.at<float>(i, 3) / kPalmInput);
        const cv::Point2f first((center.x - size.x * 0.5F) * scale - bias.x,
                                (center.y - size.y * 0.5F) * scale - bias.y);
        const cv::Point2f second((center.x + size.x * 0.5F) * scale - bias.x,
                                 (center.y + size.y * 0.5F) * scale - bias.y);
        candidateBoxes.emplace_back(cvRound(first.x), cvRound(first.y),
                                    std::max(1, cvRound(second.x - first.x)),
                                    std::max(1, cvRound(second.y - first.y)));
        candidateScores.push_back(confidence);
        sourceIndices.push_back(i);
    }
    std::vector<int> kept;
    cv::dnn::NMSBoxes(candidateBoxes, candidateScores, 0.25F, 0.30F, kept, 1.0F, 1);
    if (kept.empty()) return false;
    const int candidate = kept.front();
    const int index = sourceIndices[candidate];
    palm.box = cv::Rect2f(candidateBoxes[candidate]);
    palm.confidence = candidateScores[candidate];
    for (int landmark = 0; landmark < 7; ++landmark) {
        palm.landmarks[landmark] = cv::Point2f(
            (boxes.at<float>(index, 4 + landmark * 2) / kPalmInput + palmAnchors_[index].x) * scale - bias.x,
            (boxes.at<float>(index, 5 + landmark * 2) / kPalmInput + palmAnchors_[index].y) * scale - bias.y);
    }
    return true;
}

bool DnnAnalyzer::estimateHand(const cv::Mat& frame, const Palm& palm,
                               SceneAnalysis& result) {
    const cv::Point2f wrist = palm.landmarks[0];
    const cv::Point2f middleBase = palm.landmarks[2];
    float radians = static_cast<float>(CV_PI / 2.0) -
                    std::atan2(-(middleBase.y - wrist.y), middleBase.x - wrist.x);
    radians -= static_cast<float>(2.0 * CV_PI) *
               std::floor((radians + static_cast<float>(CV_PI)) /
                          static_cast<float>(2.0 * CV_PI));
    const float angle = radians * 180.0F / static_cast<float>(CV_PI);
    const float side = std::max(palm.box.width, palm.box.height) * 3.0F;
    cv::Point2f center(palm.box.x + palm.box.width * 0.5F,
                       palm.box.y + palm.box.height * 0.1F);
    cv::Mat transform = cv::getRotationMatrix2D(center, angle, kHandInput / side);
    transform.at<double>(0, 2) += kHandInput * 0.5 - center.x;
    transform.at<double>(1, 2) += kHandInput * 0.5 - center.y;
    cv::Mat crop;
    cv::warpAffine(frame, crop, transform, cv::Size(kHandInput, kHandInput),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    cv::cvtColor(crop, crop, cv::COLOR_BGR2RGB);
    crop.convertTo(crop, CV_32FC3, 1.0 / 255.0);
    std::vector<cv::Mat> outputs = forwardAll(handPoseNet_, nhwcFloat(crop));
    if (outputs.size() < 4) return false;

    cv::Mat screenLandmarks;
    float confidence = 0.0F;
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        if (outputs[i].total() == 63 && screenLandmarks.empty()) {
            screenLandmarks = outputs[i].reshape(1, 21);
        } else if (outputs[i].total() == 1) {
            const float value = outputs[i].at<float>(0);
            if (value > confidence) confidence = value;
        }
    }
    if (screenLandmarks.empty() || confidence < 0.80F) return false;

    cv::Mat inverse;
    cv::invertAffineTransform(transform, inverse);
    result.handLandmarks.reserve(21);
    for (int i = 0; i < 21; ++i) {
        const double x = screenLandmarks.at<float>(i, 0);
        const double y = screenLandmarks.at<float>(i, 1);
        result.handLandmarks.emplace_back(
            static_cast<float>(inverse.at<double>(0, 0) * x + inverse.at<double>(0, 1) * y + inverse.at<double>(0, 2)),
            static_cast<float>(inverse.at<double>(1, 0) * x + inverse.at<double>(1, 1) * y + inverse.at<double>(1, 2)),
            screenLandmarks.at<float>(i, 2) * side / kHandInput);
    }
    float minX = static_cast<float>(frame.cols), minY = static_cast<float>(frame.rows);
    float maxX = 0.0F, maxY = 0.0F;
    for (const auto& point : result.handLandmarks) {
        minX = std::min(minX, point.x); minY = std::min(minY, point.y);
        maxX = std::max(maxX, point.x); maxY = std::max(maxY, point.y);
    }
    result.handBox = clipped(cv::Rect(cvFloor(minX), cvFloor(minY),
                                      std::max(1, cvCeil(maxX - minX)),
                                      std::max(1, cvCeil(maxY - minY))), frame.size());
    result.handConfidence = confidence;
    result.gesture = classifyGesture(result.handLandmarks);
    return true;
}

const std::string& DnnAnalyzer::backendName() const noexcept { return backendName_; }
bool DnnAnalyzer::usingCuda() const noexcept { return cuda_; }

}  // namespace cat_meme
