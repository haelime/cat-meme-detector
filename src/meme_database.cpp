#include "cat_meme/meme_database.hpp"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include <opencv2/imgcodecs.hpp>

#include "cat_meme/image_utils.hpp"

namespace cat_meme {
namespace {

std::string lowerExtension(const std::filesystem::path& path) {
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    return extension;
}

}  // namespace

MemeDatabase::MemeDatabase(FeatureExtractor& extractor) : extractor_(extractor) {}

double weightedMatchScore(double visual, double expression, double gesture,
                          bool handDetected) noexcept {
    if (handDetected) {
        return 0.10 * visual + 0.45 * expression + 0.45 * gesture;
    }
    return 0.15 * visual + 0.85 * expression;
}

std::size_t MemeDatabase::load(const std::filesystem::path& directory,
                               const ProgressCallback& progress,
                               const SemanticAnalysisCallback& analyzeSemantic) {
    if (!std::filesystem::is_directory(directory)) {
        throw std::runtime_error("Meme directory does not exist: " + directory.string());
    }

    std::vector<std::filesystem::path> paths;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(directory)) {
        if (entry.is_regular_file() && isSupportedImage(entry.path())) {
            paths.push_back(entry.path());
        }
    }
    std::sort(paths.begin(), paths.end());
    if (progress) {
        progress(0, paths.size(), {});
    }

    std::unordered_map<std::string, SemanticProfile> labels;
    const auto labelPath = directory.parent_path() / "meme_labels.csv";
    for (const auto& [filename, profile] : loadMemeLabels(labelPath)) {
        labels.emplace(filename, profile);
    }

    std::vector<Meme> loaded;
    for (std::size_t index = 0; index < paths.size(); ++index) {
        const auto& path = paths[index];
        cv::Mat preview = loadFirstFrame(path);
        if (preview.empty()) {
            std::cerr << "Skipping unreadable image: " << path << '\n';
            if (progress) {
                progress(index + 1, paths.size(), path);
            }
            continue;
        }

        const cv::Rect center = centeredSubjectRect(preview.size(), 0.82);
        cv::Mat featureImage = preview(center);
        VisualFeature feature = extractor_.extract(featureImage);
        if (feature.colorHistogram.empty()) {
            if (progress) {
                progress(index + 1, paths.size(), path);
            }
            continue;
        }

        const auto label = labels.find(path.filename().string());
        const bool hasAuthoredLabel = label != labels.end();
        SemanticProfile semantic = !hasAuthoredLabel
                                       ? SemanticProfile{}
                                       : label->second;
        if (analyzeSemantic) {
            const SceneAnalysis inferred = analyzeSemantic(preview);
            if (semantic.expression == Expression::Unknown &&
                inferred.expression != Expression::Unknown) {
                semantic.expression = inferred.expression;
            }
            if (!hasAuthoredLabel && semantic.gesture == Gesture::None &&
                inferred.gesture != Gesture::None) {
                semantic.gesture = inferred.gesture;
            }
        }
        loaded.push_back({path, path.stem().string(), std::move(preview),
                          std::move(feature), semantic,
                          lowerExtension(path) == ".gif"});
        if (progress) {
            progress(index + 1, paths.size(), path);
        }
    }

    if (loaded.empty()) {
        throw std::runtime_error("No readable JPG, PNG, WEBP, BMP, or GIF memes in: " +
                                 directory.string());
    }

    memes_ = std::move(loaded);
    return memes_.size();
}

Match MemeDatabase::bestMatch(const cv::Mat& subject,
                              const SceneAnalysis* analysis) const {
    Match best;
    const VisualFeature query = extractor_.extract(subject);
    const bool noHandInput = analysis != nullptr &&
                             analysis->gesture == Gesture::None;
    const bool hasNoHandFallback = noHandInput && std::any_of(
        memes_.begin(), memes_.end(), [](const Meme& meme) {
            return meme.semantic.gesture == Gesture::None;
        });
    for (const auto& meme : memes_) {
        if (hasNoHandFallback && meme.semantic.gesture != Gesture::None) {
            continue;
        }
        SimilarityBreakdown score = extractor_.compare(query, meme.feature);
        const double visual = score.total;
        double expression = 0.0;
        double gesture = 0.0;
        if (analysis != nullptr && analysis->expression != Expression::Unknown) {
            expression = cat_meme::expressionScore(*analysis, meme.semantic.expression);
            if (analysis->gesture == Gesture::None) {
                score.total = weightedMatchScore(visual, expression, 0.0, false);
            } else {
                gesture = cat_meme::gestureScore(*analysis, meme.semantic.gesture);
                score.total = weightedMatchScore(visual, expression, gesture, true);
            }
        }
        if (best.meme == nullptr || score.total > best.similarity.total) {
            best = {&meme, score, visual, expression, gesture};
        }
    }
    return best;
}

const std::vector<Meme>& MemeDatabase::memes() const noexcept {
    return memes_;
}

bool isSupportedImage(const std::filesystem::path& path) {
    const std::string extension = lowerExtension(path);
    return extension == ".jpg" || extension == ".jpeg" || extension == ".png" ||
           extension == ".webp" || extension == ".bmp" || extension == ".gif";
}

cv::Mat loadFirstFrame(const std::filesystem::path& path) {
    cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR);
    if (!image.empty()) {
        return image;
    }

    if (lowerExtension(path) == ".gif") {
        cv::Animation animation;
        if (cv::imreadanimation(path.string(), animation) && !animation.frames.empty()) {
            image = normalizeBgr8(animation.frames.front());
        }
    }
    return image;
}

std::vector<std::pair<std::string, SemanticProfile>> loadMemeLabels(
    const std::filesystem::path& path) {
    std::vector<std::pair<std::string, SemanticProfile>> labels;
    if (!std::filesystem::is_regular_file(path)) {
        return labels;
    }
    std::ifstream input(path);
    std::string line;
    std::size_t lineNumber = 0;
    while (std::getline(input, line)) {
        ++lineNumber;
        if (line.empty() || line[0] == '#') continue;
        std::istringstream row(line);
        std::string filename;
        std::string expression;
        std::string gesture;
        if (!std::getline(row, filename, ',') || !std::getline(row, expression, ',') ||
            !std::getline(row, gesture)) {
            throw std::runtime_error("Invalid meme_labels.csv row " +
                                     std::to_string(lineNumber));
        }
        labels.emplace_back(filename,
                            SemanticProfile{parseExpression(expression),
                                            parseGesture(gesture)});
    }
    return labels;
}

}  // namespace cat_meme
