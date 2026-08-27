#include <algorithm>
#include <chrono>
#include <ctime>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "cat_meme/display_logic.hpp"
#include "cat_meme/face_detector.hpp"
#include "cat_meme/feature_extractor.hpp"
#include "cat_meme/hand_gesture_stabilizer.hpp"
#include "cat_meme/image_utils.hpp"
#include "cat_meme/meme_database.hpp"
#include "cat_meme/semantic_analysis.hpp"

namespace fs = std::filesystem;

namespace {

constexpr int kCanvasWidth = 1280;
constexpr int kCanvasHeight = 720;
constexpr int kPanelSize = 610;

struct Options {
    fs::path memeDirectory;
    fs::path cascadePath;
    std::optional<fs::path> inputImage;
    int cameraIndex = 0;
    double threshold = 40.0;
    bool checkAssets = false;
    bool forceCpu = false;
};

void printUsage(const char* executable) {
    std::cout
        << "Cat Meme Detector\n\n"
        << "Usage: " << executable << " [options]\n"
        << "  --camera N       Webcam index (default: 0)\n"
        << "  --memes PATH     Meme image directory\n"
        << "  --cascade PATH   OpenCV human-face cascade XML\n"
        << "  --threshold N    Minimum similarity from 0 to 100 (default: 40)\n"
        << "  --image PATH     Analyze one image instead of a webcam\n"
        << "  --check-assets   Index authored memes, then exit\n"
        << "  --cpu            Disable CUDA DNN and use the CPU\n"
        << "  --help            Show this message\n\n"
        << "Keys: Q/ESC quit, R reload memes\n";
}

Options parseOptions(int argc, char** argv) {
    const fs::path bundledAssets = fs::absolute(fs::path(argv[0])).parent_path() / "assets";
    const fs::path assets = fs::is_directory(bundledAssets)
                                ? bundledAssets
                                : fs::path(CAT_MEME_SOURCE_ASSETS);
    Options options{assets / "memes",
                    assets / "models" / "haarcascade_frontalface_alt2.xml"};

    auto valueAfter = [&](int& index, const std::string& option) -> std::string {
        if (index + 1 >= argc) {
            throw std::runtime_error("Missing value after " + option);
        }
        return argv[++index];
    };

    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--help" || argument == "-h") {
            printUsage(argv[0]);
            std::exit(0);
        } else if (argument == "--camera") {
            options.cameraIndex = std::stoi(valueAfter(i, argument));
        } else if (argument == "--memes") {
            options.memeDirectory = valueAfter(i, argument);
        } else if (argument == "--cascade") {
            options.cascadePath = valueAfter(i, argument);
        } else if (argument == "--threshold") {
            options.threshold = std::stod(valueAfter(i, argument));
        } else if (argument == "--image") {
            options.inputImage = valueAfter(i, argument);
        } else if (argument == "--check-assets") {
            options.checkAssets = true;
        } else if (argument == "--cpu") {
            options.forceCpu = true;
        } else {
            throw std::runtime_error("Unknown option: " + argument);
        }
    }

    if (options.threshold < 0.0 || options.threshold > 100.0) {
        throw std::runtime_error("--threshold must be between 0 and 100");
    }
    return options;
}

cv::Mat fitted(const cv::Mat& source, const cv::Size& size) {
    cv::Mat panel(size, CV_8UC3, cv::Scalar(19, 22, 28));
    const cv::Mat displaySource = cat_meme::normalizeBgr8(source);
    if (displaySource.empty()) {
        return panel;
    }
    const double scale = std::min(static_cast<double>(size.width) / displaySource.cols,
                                  static_cast<double>(size.height) / displaySource.rows);
    cv::Mat resized;
    cv::resize(displaySource, resized, cv::Size(), scale, scale,
               scale < 1.0 ? cv::INTER_AREA : cv::INTER_LINEAR);
    const int x = (size.width - resized.cols) / 2;
    const int y = (size.height - resized.rows) / 2;
    resized.copyTo(panel(cv::Rect(x, y, resized.cols, resized.rows)));
    return panel;
}

void putLabel(cv::Mat& image, const std::string& text, cv::Point origin,
              double scale = 0.65, const cv::Scalar& color = cv::Scalar(240, 240, 240)) {
    cv::putText(image, text, origin, cv::FONT_HERSHEY_SIMPLEX, scale,
                cv::Scalar(0, 0, 0), 4, cv::LINE_AA);
    cv::putText(image, text, origin, cv::FONT_HERSHEY_SIMPLEX, scale,
                color, 1, cv::LINE_AA);
}

void printConsoleProgress(const std::string& label, std::size_t current,
                          std::size_t total, const std::string& item) {
    constexpr std::size_t kBarWidth = 30;
    const std::size_t filled = total == 0 ? kBarWidth : current * kBarWidth / total;
    std::string displayItem = item.empty() ? "starting..." : item;
    if (displayItem.size() > 34) {
        displayItem = displayItem.substr(0, 31) + "...";
    }

    std::cout << '\r' << label << " ["
              << std::string(filled, '#')
              << std::string(kBarWidth - filled, '-') << "] "
              << current << '/' << total << "  " << std::left << std::setw(34)
              << displayItem << std::right << std::flush;
    if (current == total) {
        std::cout << '\n';
    }
}

void printPreprocessingProgress(std::size_t current, std::size_t total,
                                const fs::path& path) {
    printConsoleProgress("Analyzing memes   ", current, total,
                         path.empty() ? std::string() : path.filename().string());
}

void printModelLoadingProgress(std::size_t current, std::size_t total,
                               const fs::path& path) {
    printConsoleProgress("Loading AI models ", current, total,
                         path.empty() ? std::string() : path.filename().string());
}

class MemePlayer {
public:
    const cv::Mat& frameFor(const cat_meme::Meme* meme) {
        if (meme != current_) {
            select(meme);
        }
        if (current_ == nullptr || animation_.frames.empty()) {
            return frame_;
        }

        const auto now = std::chrono::steady_clock::now();
        if (now >= nextFrame_) {
            cv::Mat normalized = cat_meme::normalizeBgr8(animation_.frames[frameIndex_]);
            if (!normalized.empty()) {
                frame_ = std::move(normalized);
            }
            const int duration = frameIndex_ < animation_.durations.size()
                                     ? animation_.durations[frameIndex_]
                                     : 83;
            nextFrame_ = now + std::chrono::milliseconds(std::max(20, duration));
            frameIndex_ = (frameIndex_ + 1) % animation_.frames.size();
        }
        return frame_;
    }

private:
    void select(const cat_meme::Meme* meme) {
        animation_ = cv::Animation();
        frameIndex_ = 0;
        current_ = meme;
        frame_ = meme == nullptr ? cv::Mat() : meme->preview;
        nextFrame_ = std::chrono::steady_clock::now();

        if (meme != nullptr && meme->animated) {
            cv::imreadanimation(meme->path.string(), animation_);
        }
    }

    const cat_meme::Meme* current_ = nullptr;
    cv::Animation animation_;
    std::size_t frameIndex_ = 0;
    cv::Mat frame_;
    std::chrono::steady_clock::time_point nextFrame_{};
};

cv::Mat compose(const cv::Mat& cameraFrame, const cv::Rect& subject,
                bool faceDetected, const cat_meme::Match& match,
                const cat_meme::SceneAnalysis& analysis,
                const std::string& backend, double threshold, MemePlayer& player) {
    cv::Mat canvas(kCanvasHeight, kCanvasWidth, CV_8UC3, cv::Scalar(12, 14, 19));
    cv::Mat annotated = cameraFrame.clone();
    const cv::Scalar subjectColor = faceDetected ? cv::Scalar(70, 230, 120)
                                                 : cv::Scalar(40, 170, 255);
    cv::rectangle(annotated, subject, subjectColor, 3, cv::LINE_AA);
    if (!analysis.handBox.empty()) {
        static constexpr int connections[][2] = {
            {0,1},{1,2},{2,3},{3,4}, {0,5},{5,6},{6,7},{7,8},
            {0,9},{9,10},{10,11},{11,12}, {0,13},{13,14},{14,15},{15,16},
            {0,17},{17,18},{18,19},{19,20}, {5,9},{9,13},{13,17}
        };
        cv::rectangle(annotated, analysis.handBox, cv::Scalar(255, 180, 70), 2,
                      cv::LINE_AA);
        for (const auto& connection : connections) {
            if (analysis.handLandmarks.size() <=
                static_cast<std::size_t>(std::max(connection[0], connection[1]))) continue;
            const auto& first = analysis.handLandmarks[connection[0]];
            const auto& second = analysis.handLandmarks[connection[1]];
            cv::line(annotated, cv::Point(cvRound(first.x), cvRound(first.y)),
                     cv::Point(cvRound(second.x), cvRound(second.y)),
                     cv::Scalar(255, 180, 70), 2, cv::LINE_AA);
        }
        for (const auto& point : analysis.handLandmarks) {
            cv::circle(annotated, cv::Point(cvRound(point.x), cvRound(point.y)), 3,
                       cv::Scalar(70, 240, 255), -1, cv::LINE_AA);
        }
    }

    fitted(annotated, cv::Size(kPanelSize, kPanelSize))
        .copyTo(canvas(cv::Rect(15, 75, kPanelSize, kPanelSize)));
    putLabel(canvas, "WEBCAM", cv::Point(15, 48), 0.8, cv::Scalar(120, 230, 255));
    putLabel(canvas, "BEST MEME", cv::Point(655, 48), 0.8, cv::Scalar(120, 230, 255));
    putLabel(canvas, faceDetected ? "HUMAN FACE" : "CENTER SCAN",
             cv::Point(30, 105), 0.58, subjectColor);
    std::ostringstream state;
    state << "EXPR " << cat_meme::toString(analysis.expression)
          << "  HAND " << cat_meme::toString(analysis.gesture)
          << "  DNN " << backend;
    putLabel(canvas, state.str(), cv::Point(30, 700), 0.52,
             cv::Scalar(180, 205, 230));

    const bool visible = cat_meme::shouldDisplayMeme(
        faceDetected, match.meme != nullptr, match.similarity.total, threshold);
    if (visible) {
        fitted(player.frameFor(match.meme), cv::Size(kPanelSize, kPanelSize))
            .copyTo(canvas(cv::Rect(655, 75, kPanelSize, kPanelSize)));
        std::ostringstream label;
        const bool confident = match.similarity.total >= threshold;
        label << (confident ? "MATCH  " : "CLOSEST  ") << match.meme->displayName
              << "  " << std::fixed << std::setprecision(1)
              << match.similarity.total << "%";
        putLabel(canvas, label.str(), cv::Point(670, 670), 0.62,
                 confident ? cv::Scalar(90, 245, 140) : cv::Scalar(40, 170, 255));
    } else {
        player.frameFor(nullptr);
        putLabel(canvas, "No similar meme yet",
                 cv::Point(810, 365), 0.8, cv::Scalar(150, 160, 175));
    }
    return canvas;
}

cat_meme::Match matchFrame(const cv::Mat& frame, const cv::Rect& face,
                           const cat_meme::MemeDatabase& database,
                           const cat_meme::SceneAnalysis* analysis = nullptr) {
    const cv::Rect crop = cat_meme::paddedRect(face, frame.size());
    return database.bestMatch(frame(crop), analysis);
}

cat_meme::SceneAnalysis analyzeMemePreview(const cv::Mat& preview,
                                           cat_meme::FaceDetector& detector,
                                           cat_meme::DnnAnalyzer& analyzer) {
    const auto face = detector.largestFace(preview);
    const cv::Rect expressionArea = face.value_or(
        cat_meme::centeredSubjectRect(preview.size(), 0.82));
    return analyzer.analyze(preview, expressionArea);
}

int runStillImage(const fs::path& path, cat_meme::FaceDetector& detector,
                  cat_meme::DnnAnalyzer& analyzer,
                  const cat_meme::MemeDatabase& database, double threshold) {
    const cv::Mat image = cat_meme::loadFirstFrame(path);
    if (image.empty()) {
        throw std::runtime_error("Could not read input image: " + path.string());
    }
    const auto face = detector.largestFace(image);
    const cv::Rect subject = face.value_or(cat_meme::centeredSubjectRect(image.size()));
    if (!face) {
        std::cerr << "Human-face cascade missed; using the center scan area\n";
    }
    const cat_meme::SceneAnalysis analysis = analyzer.analyze(
        image, face.value_or(cv::Rect()));
    const cat_meme::Match match = matchFrame(image, subject, database, &analysis);
    MemePlayer player;
    cv::imshow("Cat Meme Detector",
               compose(image, subject, face.has_value(), match, analysis,
                       analyzer.backendName(), threshold, player));
    std::cout << "Best match: " << match.meme->path << " ("
              << std::fixed << std::setprecision(1) << match.similarity.total << "%)\n";
    cv::waitKey(0);
    return 0;
}

int runWebcam(const Options& options, cat_meme::FaceDetector& detector,
              cat_meme::DnnAnalyzer& analyzer, cat_meme::MemeDatabase& database,
              const std::function<void(const std::string&)>& appendLog) {
    cv::VideoCapture camera(options.cameraIndex, cv::CAP_ANY);
    if (!camera.isOpened()) {
        throw std::runtime_error("Could not open webcam index " +
                                 std::to_string(options.cameraIndex));
    }
    camera.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    camera.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    cv::namedWindow("Cat Meme Detector", cv::WINDOW_AUTOSIZE);
    MemePlayer player;
    cat_meme::Match match;
    cat_meme::SceneAnalysis analysis;
    // A gesture affects matching only after several consistent DNN observations.
    cat_meme::HandGestureStabilizer handStabilizer;
    int frameNumber = 0;
    const cat_meme::Meme* lastLoggedMeme = nullptr;
    bool lastLoggedCatDetection = false;

    while (true) {
        cv::Mat frame;
        if (!camera.read(frame) || frame.empty()) {
            throw std::runtime_error("Webcam stopped returning frames");
        }
        cv::flip(frame, frame, 1);
        const auto face = detector.largestFace(frame);
        const cv::Rect subject = face.value_or(cat_meme::centeredSubjectRect(frame.size()));
        if (frameNumber % 5 == 0) {
            analysis = analyzer.analyze(frame, face.value_or(cv::Rect()));
            analysis.gesture = handStabilizer.update(analysis.gesture);
            if (analysis.gesture == cat_meme::Gesture::None) {
                analysis.handConfidence = 0.0F;
                analysis.handBox = {};
                analysis.handLandmarks.clear();
            }
            match = matchFrame(frame, subject, database, &analysis);
            if (match.meme != lastLoggedMeme || face.has_value() != lastLoggedCatDetection) {
                std::ostringstream message;
                message << "Closest meme=" << match.meme->displayName
                        << " similarity=" << std::fixed << std::setprecision(1)
                        << match.similarity.total
                        << " human_face=" << (face ? "yes" : "no")
                        << " expression=" << cat_meme::toString(analysis.expression)
                        << " gesture=" << cat_meme::toString(analysis.gesture)
                        << " backend=" << analyzer.backendName()
                        << " display="
                        << (cat_meme::shouldDisplayMeme(face.has_value(), true,
                                                       match.similarity.total,
                                                       options.threshold)
                                ? "yes"
                                : "no");
                appendLog(message.str());
                lastLoggedMeme = match.meme;
                lastLoggedCatDetection = face.has_value();
            }
        }

        cv::imshow("Cat Meme Detector",
                   compose(frame, subject, face.has_value(), match,
                           analysis, analyzer.backendName(), options.threshold, player));
        const int key = cv::waitKey(1) & 0xFF;
        if (key == 27 || key == 'q' || key == 'Q') {
            break;
        }
        if (key == 'r' || key == 'R') {
            std::cout << "Reloaded "
                      << database.load(
                             options.memeDirectory, printPreprocessingProgress,
                             [&](const cv::Mat& preview) {
                                 return analyzeMemePreview(preview, detector, analyzer);
                             })
                      << " memes\n";
            match = {};
        }
        ++frameNumber;
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const fs::path logPath = fs::absolute(fs::path(argv[0])).parent_path() /
                             "cat_meme_detector.log";
    auto appendLog = [&](const std::string& message) {
        std::ofstream output(logPath, std::ios::app);
        const std::time_t now = std::chrono::system_clock::to_time_t(
            std::chrono::system_clock::now());
        std::tm localTime{};
#ifdef _WIN32
        localtime_s(&localTime, &now);
#else
        localtime_r(&now, &localTime);
#endif
        output << std::put_time(&localTime, "%Y-%m-%d %H:%M:%S")
               << "  " << message << '\n';
    };

    try {
        appendLog("Starting Cat Meme Detector");
        const Options options = parseOptions(argc, argv);
        printConsoleProgress("Loading face model", 0, 1, "starting...");
        cat_meme::FaceDetector detector(options.cascadePath);
        printConsoleProgress("Loading face model", 1, 1,
                             options.cascadePath.filename().string());
        cat_meme::FeatureExtractor extractor;
        cat_meme::MemeDatabase database(extractor);

        if (options.checkAssets) {
            const std::size_t indexedCount = database.load(
                options.memeDirectory, printPreprocessingProgress);
            std::cout << "Indexed " << indexedCount << " memes from "
                      << options.memeDirectory << '\n';
            appendLog("Indexed " + std::to_string(indexedCount) + " memes");
            return 0;
        }

        const fs::path modelDirectory = options.cascadePath.parent_path();
        cat_meme::DnnAnalyzer analyzer(
            modelDirectory / "facial_expression_recognition_mobilefacenet_2022july.onnx",
            modelDirectory / "palm_detection_mediapipe_2023feb.onnx",
            modelDirectory / "handpose_estimation_mediapipe_2023feb.onnx",
            !options.forceCpu, printModelLoadingProgress);
        std::cout << "DNN backend: " << analyzer.backendName() << '\n';
        appendLog("DNN backend=" + analyzer.backendName());

        const std::size_t indexedCount = database.load(
            options.memeDirectory, printPreprocessingProgress,
            [&](const cv::Mat& preview) {
                return analyzeMemePreview(preview, detector, analyzer);
            });
        std::cout << "Analyzed and indexed " << indexedCount << " memes from "
                  << options.memeDirectory << '\n';
        appendLog("Analyzed and indexed " + std::to_string(indexedCount) + " memes");

        if (options.inputImage) {
            return runStillImage(*options.inputImage, detector, analyzer,
                                 database, options.threshold);
        }
        return runWebcam(options, detector, analyzer, database, appendLog);
    } catch (const std::exception& error) {
        appendLog(std::string("Fatal error: ") + error.what());
        std::cerr << "Error: " << error.what() << "\n\n";
        printUsage(argv[0]);
        return 1;
    }
}
