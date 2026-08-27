#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_set>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "cat_meme/feature_extractor.hpp"
#include "cat_meme/display_logic.hpp"
#include "cat_meme/face_detector.hpp"
#include "cat_meme/hand_gesture_stabilizer.hpp"
#include "cat_meme/image_utils.hpp"
#include "cat_meme/meme_database.hpp"
#include "cat_meme/semantic_analysis.hpp"

namespace fs = std::filesystem;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

class TemporaryDirectory {
public:
    explicit TemporaryDirectory(const std::string& prefix) {
        const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = fs::temp_directory_path() / (prefix + std::to_string(suffix));
        fs::create_directories(path_);
    }

    ~TemporaryDirectory() {
        std::error_code error;
        fs::remove_all(path_, error);
    }

    [[nodiscard]] const fs::path& path() const noexcept { return path_; }

private:
    fs::path path_;
};

cv::Mat testPattern(const cv::Scalar& base) {
    cv::Mat image(256, 256, CV_8UC3, base);
    cv::circle(image, cv::Point(128, 115), 78, cv::Scalar(210, 190, 160), -1);
    cv::circle(image, cv::Point(98, 100), 13, cv::Scalar(15, 20, 20), -1);
    cv::circle(image, cv::Point(158, 100), 13, cv::Scalar(15, 20, 20), -1);
    cv::line(image, cv::Point(110, 155), cv::Point(128, 165), cv::Scalar(20, 20, 20), 5);
    cv::line(image, cv::Point(128, 165), cv::Point(146, 155), cv::Scalar(20, 20, 20), 5);
    return image;
}

void testFeatureSimilarity() {
    cat_meme::FeatureExtractor extractor;
    const cv::Mat original = testPattern(cv::Scalar(30, 80, 160));
    const cv::Mat different = testPattern(cv::Scalar(170, 45, 20));
    cv::Mat shifted;
    const cv::Mat translation = (cv::Mat_<double>(2, 3) << 1, 0, 4, 0, 1, -3);
    cv::warpAffine(original, shifted, translation, original.size());

    const auto originalFeature = extractor.extract(original);
    const auto same = extractor.compare(originalFeature, extractor.extract(original));
    const auto near = extractor.compare(originalFeature, extractor.extract(shifted));
    const auto far = extractor.compare(originalFeature, extractor.extract(different));
    require(same.total > 99.0, "An identical image must score near 100");
    require(near.total > far.total, "A small translation must beat a recolored image");
}

void testAuthoredAssetHierarchy() {
    const fs::path source = CAT_MEME_SOURCE_DIR;
    const fs::path memes = source / "assets" / "memes";
    const fs::path cascade = source / "assets" / "models" /
                             "haarcascade_frontalface_alt2.xml";
    const fs::path obsoleteCatCascade = source / "assets" / "models" /
                                        "haarcascade_frontalcatface_extended.xml";
    const fs::path labels = source / "assets" / "meme_labels.csv";
    require(fs::is_directory(memes), "The authored assets/memes hierarchy is missing");
    require(fs::is_regular_file(cascade), "The authored human-face model is missing");
    require(!fs::exists(obsoleteCatCascade), "The obsolete cat detector model still exists");
    require(fs::is_regular_file(labels), "The authored meme semantic labels are missing");
    require(cat_meme::isSupportedImage("meme.webp") &&
                cat_meme::isSupportedImage("meme.WEBP"),
            "WEBP files must be accepted case-insensitively");
    for (const char* model : {
             "facial_expression_recognition_mobilefacenet_2022july.onnx",
             "palm_detection_mediapipe_2023feb.onnx",
             "handpose_estimation_mediapipe_2023feb.onnx"}) {
        require(fs::file_size(source / "assets" / "models" / model) > 1000000,
                std::string("The authored DNN model is missing: ") + model);
    }

    bool foundImage = false;
    for (const auto& entry : fs::directory_iterator(memes)) {
        foundImage = foundImage || (entry.is_regular_file() &&
                                    cat_meme::isSupportedImage(entry.path()));
    }
    require(foundImage, "The authored meme hierarchy contains no images");
}

void testBundledAssetsMirrorAuthoredHierarchy() {
    const fs::path source = fs::path(CAT_MEME_SOURCE_DIR) / "assets" / "memes";
    const fs::path bundled = fs::path(CAT_MEME_BINARY_DIR) / "assets" / "memes";
    std::unordered_set<std::string> sourceFiles;
    std::unordered_set<std::string> bundledFiles;
    for (const auto& entry : fs::recursive_directory_iterator(source)) {
        if (entry.is_regular_file() && cat_meme::isSupportedImage(entry.path())) {
            sourceFiles.insert(fs::relative(entry.path(), source).generic_string());
        }
    }
    for (const auto& entry : fs::recursive_directory_iterator(bundled)) {
        if (entry.is_regular_file() && cat_meme::isSupportedImage(entry.path())) {
            bundledFiles.insert(fs::relative(entry.path(), bundled).generic_string());
        }
    }
    require(sourceFiles == bundledFiles,
            "Bundled memes must exactly mirror the authored meme hierarchy");
}

std::vector<cv::Point3f> syntheticHand(bool thumb, bool index, bool middle,
                                       bool ring, bool pinky) {
    std::vector<cv::Point3f> points(21, cv::Point3f());
    const auto finger = [&](int joint, int tip, float x, bool open) {
        points[joint] = cv::Point3f(x, -10.0F, 0.0F);
        points[tip] = cv::Point3f(x, open ? -24.0F : -6.0F, 0.0F);
    };
    points[3] = cv::Point3f(10.0F, 0.0F, 0.0F);
    points[4] = cv::Point3f(thumb ? 24.0F : 6.0F, 0.0F, 0.0F);
    finger(6, 8, -7.0F, index);
    finger(10, 12, -2.0F, middle);
    finger(14, 16, 3.0F, ring);
    finger(18, 20, 8.0F, pinky);
    return points;
}

void testGestureClassification() {
    require(cat_meme::classifyGesture(syntheticHand(false, false, false, false, false)) ==
                cat_meme::Gesture::Fist,
            "Closed landmarks must classify as fist");
    require(cat_meme::classifyGesture(syntheticHand(true, true, true, true, true)) ==
                cat_meme::Gesture::OpenPalm,
            "Extended landmarks must classify as open palm");
    require(cat_meme::classifyGesture(syntheticHand(false, true, true, false, false)) ==
                cat_meme::Gesture::Peace,
            "Two extended fingers must classify as peace");
    require(cat_meme::classifyGesture(syntheticHand(false, true, false, false, false)) ==
                cat_meme::Gesture::Pointing,
            "One extended index finger must classify as pointing");
    require(cat_meme::classifyGesture(syntheticHand(true, false, false, false, false)) ==
                cat_meme::Gesture::ThumbsUp,
            "An isolated thumb must classify as thumbs up");
}

void testHandGestureHasEqualSemanticWeight() {
    const double gestureOnly = cat_meme::weightedMatchScore(0.0, 0.0, 100.0, true);
    const double expressionOnly = cat_meme::weightedMatchScore(0.0, 100.0, 0.0, true);
    const double visualOnly = cat_meme::weightedMatchScore(100.0, 0.0, 0.0, true);
    require(std::abs(gestureOnly - 45.0) < 0.001,
            "A detected hand gesture must contribute 45 percent of the match score");
    require(std::abs(gestureOnly - expressionOnly) < 0.001,
            "Hand gesture and expression must have equal semantic weight");
    require(gestureOnly > visualOnly,
            "Hand gesture must outweigh visual similarity when a hand is detected");
}

void testHandGestureStabilizerRejectsTransientFalsePositives() {
    cat_meme::HandGestureStabilizer stabilizer(4, 2);
    for (const auto observed : {
             cat_meme::Gesture::Pointing,
             cat_meme::Gesture::None,
             cat_meme::Gesture::ThumbsUp,
             cat_meme::Gesture::Fist,
             cat_meme::Gesture::Other,
             cat_meme::Gesture::None}) {
        require(stabilizer.update(observed) == cat_meme::Gesture::None,
                "Transient hand false positives must remain suppressed");
    }
}

void testHandGestureStabilizerConfirmsAndReleasesRealGesture() {
    cat_meme::HandGestureStabilizer stabilizer(4, 2);
    for (int observation = 0; observation < 3; ++observation) {
        require(stabilizer.update(cat_meme::Gesture::Pointing) ==
                    cat_meme::Gesture::None,
                "A hand gesture must not activate before four confirmations");
    }
    require(stabilizer.update(cat_meme::Gesture::Pointing) ==
                cat_meme::Gesture::Pointing,
            "Four matching observations must confirm the hand gesture");
    require(stabilizer.update(cat_meme::Gesture::None) ==
                cat_meme::Gesture::Pointing,
            "One missed frame must not release a confirmed hand gesture");
    require(stabilizer.update(cat_meme::Gesture::None) ==
                cat_meme::Gesture::None,
            "Two missed frames must release a confirmed hand gesture");
}

void testAuthoredSemanticLabelsAreValidOverrides() {
    const fs::path source = CAT_MEME_SOURCE_DIR;
    const auto labels = cat_meme::loadMemeLabels(source / "assets" / "meme_labels.csv");
    require(!labels.empty(), "The authored semantic override file is empty");
    std::unordered_set<std::string> filenames;
    for (const auto& [filename, profile] : labels) {
        require(!filename.empty(), "A semantic override has an empty filename");
        require(filenames.insert(filename).second,
                "A meme has duplicate semantic overrides: " + filename);
        require(profile.expression != cat_meme::Expression::Unknown,
                "An authored semantic override must specify an expression: " + filename);
    }
}

void testDnnModelsLoadAndInferOnCpu() {
    const fs::path models = fs::path(CAT_MEME_SOURCE_DIR) / "assets" / "models";
    std::size_t modelProgressCalls = 0;
    cat_meme::DnnAnalyzer analyzer(
        models / "facial_expression_recognition_mobilefacenet_2022july.onnx",
        models / "palm_detection_mediapipe_2023feb.onnx",
        models / "handpose_estimation_mediapipe_2023feb.onnx", false,
        [&](std::size_t current, std::size_t total, const fs::path& model) {
            require(current == modelProgressCalls,
                    "Model loading progress must start at zero and be sequential");
            require(total == 3, "Model loading progress total is incorrect");
            require(current == 0 || !model.empty(),
                    "Completed model loading progress must name its model");
            ++modelProgressCalls;
        });
    require(modelProgressCalls == 4,
            "Model loading progress must report start and all three models");
    cv::Mat frame(240, 320, CV_8UC3, cv::Scalar(110, 125, 140));
    cv::circle(frame, cv::Point(160, 120), 70, cv::Scalar(170, 180, 195), -1);
    const auto analysis = analyzer.analyze(frame, cv::Rect(90, 50, 140, 140));
    require(analysis.expression != cat_meme::Expression::Unknown,
            "The facial-expression ONNX model did not produce an output");
    require(analysis.expressionConfidence > 0.0F,
            "The facial-expression confidence must be positive");
    require(!analyzer.usingCuda() && analyzer.backendName() == "CPU",
            "The explicit CPU DNN smoke test selected the wrong backend");

    const cv::Mat handSample = cv::imread(
        (fs::path(CAT_MEME_SOURCE_DIR) / "tests" / "data" /
         "opencv_zoo_gesture_classification.png").string());
    require(!handSample.empty(), "The authored hand DNN test image is missing");
    const cv::Mat oneFinger = handSample(cv::Rect(262, 0, 262, 196)).clone();
    const auto handAnalysis = analyzer.analyze(oneFinger, cv::Rect());
    require(handAnalysis.handLandmarks.size() == 21 && !handAnalysis.handBox.empty(),
            "Palm detection did not feed 21 landmarks into hand-pose inference");
    require(handAnalysis.gesture != cat_meme::Gesture::None,
            "The hand landmarks did not produce a gesture");
}

void testEveryAuthoredMemeIsIndexable() {
    const fs::path source = CAT_MEME_SOURCE_DIR;
    const fs::path memes = source / "assets" / "memes";

    std::size_t authoredCount = 0;
    for (const auto& entry : fs::recursive_directory_iterator(memes)) {
        if (entry.is_regular_file() && cat_meme::isSupportedImage(entry.path())) {
            ++authoredCount;
        }
    }

    cat_meme::FeatureExtractor extractor;
    cat_meme::MemeDatabase database(extractor);
    std::size_t progressCalls = 0;
    const std::size_t indexedCount = database.load(
        memes, [&](std::size_t current, std::size_t total, const fs::path&) {
            ++progressCalls;
            require(current + 1 == progressCalls,
                    "Preprocessing progress must start at zero and be sequential");
            require(total == authoredCount, "Preprocessing progress total is incorrect");
        });
    require(indexedCount == authoredCount,
            "Every authored meme image must be readable and indexable");
    require(progressCalls == authoredCount + 1,
            "Preprocessing progress must report every authored meme");
}

void testRuntimeSemanticAnalysisFillsMissingLabels() {
    const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path temp = fs::temp_directory_path() /
                          ("cat_meme_semantic_test_" + std::to_string(suffix));
    fs::create_directories(temp);
    require(cv::imwrite((temp / "unlabelled_meme.png").string(),
                        testPattern(cv::Scalar(30, 80, 160))),
            "Could not author the temporary semantic-analysis fixture");

    cat_meme::FeatureExtractor extractor;
    cat_meme::MemeDatabase database(extractor);
    std::size_t analysisCalls = 0;
    database.load(temp, {}, [&](const cv::Mat& preview) {
        require(!preview.empty(), "Runtime semantic analysis received an empty meme");
        ++analysisCalls;
        cat_meme::SceneAnalysis inferred;
        inferred.expression = cat_meme::Expression::Happy;
        inferred.expressionConfidence = 0.9F;
        inferred.gesture = cat_meme::Gesture::ThumbsUp;
        inferred.handConfidence = 0.9F;
        return inferred;
    });

    require(analysisCalls == 1, "Every unlabelled meme must be analyzed at runtime");
    require(database.memes().size() == 1 &&
                database.memes().front().semantic.expression ==
                    cat_meme::Expression::Happy &&
                database.memes().front().semantic.gesture ==
                    cat_meme::Gesture::ThumbsUp,
            "Runtime analysis did not fill the missing meme semantic labels");
    fs::remove_all(temp);
}

void testExplicitNoHandLabelOverridesRuntimeInference() {
    TemporaryDirectory temp("cat_meme_no_hand_override_test_");
    const fs::path memes = temp.path() / "memes";
    fs::create_directories(memes);
    require(cv::imwrite((memes / "son.png").string(),
                        testPattern(cv::Scalar(80, 100, 120))),
            "Could not author the explicit no-hand fixture");
    {
        std::ofstream labels(temp.path() / "meme_labels.csv");
        labels << "# filename,expression,gesture\n"
               << "son.png,neutral,none\n";
    }

    cat_meme::FeatureExtractor extractor;
    cat_meme::MemeDatabase database(extractor);
    database.load(memes, {}, [](const cv::Mat&) {
        cat_meme::SceneAnalysis inferred;
        inferred.expression = cat_meme::Expression::Neutral;
        inferred.gesture = cat_meme::Gesture::Pointing;
        inferred.handConfidence = 0.99F;
        return inferred;
    });
    require(database.memes().front().semantic.gesture == cat_meme::Gesture::None,
            "An authored none gesture must remain a no-hand fallback");
}

void testNoHandInputSelectsNoHandMeme() {
    TemporaryDirectory temp("cat_meme_no_hand_match_test_");
    const fs::path memes = temp.path() / "memes";
    fs::create_directories(memes);
    const cv::Mat identical = testPattern(cv::Scalar(50, 90, 130));
    require(cv::imwrite((memes / "a_pointing.png").string(), identical) &&
                cv::imwrite((memes / "z_no_hand.png").string(), identical),
            "Could not author the no-hand matching fixtures");
    {
        std::ofstream labels(temp.path() / "meme_labels.csv");
        labels << "# filename,expression,gesture\n"
               << "a_pointing.png,neutral,pointing\n"
               << "z_no_hand.png,neutral,none\n";
    }

    cat_meme::FeatureExtractor extractor;
    cat_meme::MemeDatabase database(extractor);
    database.load(memes);
    cat_meme::SceneAnalysis input;
    input.expression = cat_meme::Expression::Neutral;
    input.expressionProbabilities[static_cast<std::size_t>(
        cat_meme::Expression::Neutral)] = 1.0F;
    input.gesture = cat_meme::Gesture::None;
    const auto match = database.bestMatch(identical, &input);
    require(match.meme != nullptr && match.meme->path.filename() == "z_no_hand.png",
            "No-hand input must fall back to a meme without a hand gesture");
}

void testCenterScanFallback() {
    const cv::Rect region = cat_meme::centeredSubjectRect(cv::Size(1280, 720));
    require(region.width == region.height, "The center scan fallback must be square");
    require(region.x >= 0 && region.y >= 0 && region.br().x <= 1280 &&
                region.br().y <= 720,
            "The center scan fallback must stay inside the camera frame");

    const fs::path cascade = fs::path(CAT_MEME_SOURCE_DIR) / "assets" / "models" /
                             "haarcascade_frontalface_alt2.xml";
    cat_meme::FaceDetector detector(cascade);
    const cv::Mat blank(720, 1280, CV_8UC3, cv::Scalar(127, 127, 127));
    require(!detector.largestFace(blank), "A blank frame must not be a human face");
}

void testAuthoredMemesProduceVisibleMatches() {
    const fs::path source = CAT_MEME_SOURCE_DIR;
    const fs::path memes = source / "assets" / "memes";
    cat_meme::FeatureExtractor extractor;
    cat_meme::MemeDatabase database(extractor);
    database.load(memes);

    std::size_t readableCount = 0;
    std::size_t visibleMatchCount = 0;
    for (const auto& entry : fs::recursive_directory_iterator(memes)) {
        if (!entry.is_regular_file() || !cat_meme::isSupportedImage(entry.path())) {
            continue;
        }
        const cv::Mat image = cat_meme::loadFirstFrame(entry.path());
        if (image.empty()) {
            continue;
        }
        ++readableCount;
        const cv::Rect subject = cat_meme::centeredSubjectRect(image.size(), 0.82);
        if (database.bestMatch(image(subject)).similarity.total >= 40.0) {
            ++visibleMatchCount;
        }
    }
    require(visibleMatchCount * 2 >= readableCount,
            "At least half of the authored memes must produce a visible default match");
}

void testAnimatedMemesDecodeRepeatedly() {
    const fs::path memes = fs::path(CAT_MEME_SOURCE_DIR) / "assets" / "memes";
    std::size_t animationCount = 0;
    for (const auto& entry : fs::recursive_directory_iterator(memes)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".gif") {
            continue;
        }
        ++animationCount;
        cv::Animation animation;
        require(cv::imreadanimation(entry.path().string(), animation),
                "Could not open animated meme: " + entry.path().string());
        require(!animation.frames.empty(),
                "Animated meme yielded no frames: " + entry.path().string());

        for (int cycle = 0; cycle < 3; ++cycle) {
            for (const cv::Mat& frame : animation.frames) {
                const cv::Mat displayFrame = cat_meme::normalizeBgr8(frame);
                require(!displayFrame.empty() && displayFrame.type() == CV_8UC3,
                        "Animated meme could not be normalized for display: " +
                            entry.path().string());
            }
        }
    }
    require(animationCount > 0, "The playback regression test found no animated memes");
}

void testDetectedFaceAlwaysDisplaysClosestMeme() {
    require(cat_meme::shouldDisplayMeme(true, true, 12.0, 40.0),
            "A detected face must display its closest meme even below the threshold");
    require(!cat_meme::shouldDisplayMeme(false, true, 12.0, 40.0),
            "A low-confidence center scan must remain hidden");
    require(cat_meme::shouldDisplayMeme(false, true, 45.0, 40.0),
            "A high-confidence center scan must display its meme");
    require(!cat_meme::shouldDisplayMeme(true, false, 100.0, 40.0),
            "No image can be displayed without a match");
}

void testNoGeneratedVisualFallback() {
    const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path temp = fs::temp_directory_path() /
                          ("cat_meme_empty_database_test_" + std::to_string(suffix));
    fs::create_directories(temp);

    cat_meme::FeatureExtractor extractor;
    cat_meme::MemeDatabase database(extractor);
    bool rejectedEmptyHierarchy = false;
    try {
        database.load(temp);
    } catch (const std::runtime_error&) {
        rejectedEmptyHierarchy = true;
    }
    require(rejectedEmptyHierarchy, "An empty authored meme hierarchy must be rejected");
    require(fs::is_empty(temp), "Runtime generated a hidden fallback visual");
    fs::remove_all(temp);
}

}  // namespace

int main() {
    try {
        testFeatureSimilarity();
        testAuthoredAssetHierarchy();
        testBundledAssetsMirrorAuthoredHierarchy();
        testEveryAuthoredMemeIsIndexable();
        testCenterScanFallback();
        testAuthoredMemesProduceVisibleMatches();
        testAnimatedMemesDecodeRepeatedly();
        testDetectedFaceAlwaysDisplaysClosestMeme();
        testGestureClassification();
        testHandGestureHasEqualSemanticWeight();
        testHandGestureStabilizerRejectsTransientFalsePositives();
        testHandGestureStabilizerConfirmsAndReleasesRealGesture();
        testAuthoredSemanticLabelsAreValidOverrides();
        testDnnModelsLoadAndInferOnCpu();
        testRuntimeSemanticAnalysisFillsMissingLabels();
        testExplicitNoHandLabelOverridesRuntimeInference();
        testNoHandInputSelectsNoHandMeme();
        testNoGeneratedVisualFallback();
        std::cout << "All cat meme tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Test failure: " << error.what() << '\n';
        return 1;
    }
}
