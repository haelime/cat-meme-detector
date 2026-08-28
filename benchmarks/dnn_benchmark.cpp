#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#endif

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "cat_meme/semantic_analysis.hpp"

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {

struct Options {
    bool useCuda = false;
    int warmup = 10;
    int iterations = 100;
};

struct CycleTiming {
    double faceMilliseconds = 0.0;
    double handMilliseconds = 0.0;
    double totalMilliseconds = 0.0;
    double checksum = 0.0;
    bool expressionProduced = false;
    std::size_t handLandmarkCount = 0;
};

struct Distribution {
    double mean = 0.0;
    double median = 0.0;
    double p95 = 0.0;
    double minimum = 0.0;
    double maximum = 0.0;
};

double milliseconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

Options parseOptions(int argc, char** argv) {
    Options options;
    auto valueAfter = [&](int& index, const std::string& option) -> std::string {
        if (index + 1 >= argc) {
            throw std::runtime_error("Missing value after " + option);
        }
        return argv[++index];
    };

    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--backend") {
            const std::string backend = valueAfter(i, argument);
            if (backend == "cuda") {
                options.useCuda = true;
            } else if (backend == "cpu") {
                options.useCuda = false;
            } else {
                throw std::runtime_error("--backend must be cpu or cuda");
            }
        } else if (argument == "--warmup") {
            options.warmup = std::stoi(valueAfter(i, argument));
        } else if (argument == "--iterations") {
            options.iterations = std::stoi(valueAfter(i, argument));
        } else {
            throw std::runtime_error("Unknown option: " + argument);
        }
    }

    if (options.warmup < 0 || options.iterations < 1) {
        throw std::runtime_error("Warmup must be non-negative and iterations must be positive");
    }
    return options;
}

Distribution summarize(std::vector<double> values) {
    if (values.empty()) return {};
    std::sort(values.begin(), values.end());
    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    const std::size_t middle = values.size() / 2;
    const double median = values.size() % 2 == 0
                              ? (values[middle - 1] + values[middle]) / 2.0
                              : values[middle];
    const std::size_t p95Index = static_cast<std::size_t>(
        std::ceil(0.95 * static_cast<double>(values.size()))) - 1;
    return {sum / static_cast<double>(values.size()), median,
            values[std::min(p95Index, values.size() - 1)],
            values.front(), values.back()};
}

#ifdef _WIN32
double fileTimeMilliseconds(const FILETIME& value) {
    ULARGE_INTEGER ticks{};
    ticks.LowPart = value.dwLowDateTime;
    ticks.HighPart = value.dwHighDateTime;
    return static_cast<double>(ticks.QuadPart) / 10000.0;
}

double processCpuMilliseconds() {
    FILETIME creation{}, exit{}, kernel{}, user{};
    if (!GetProcessTimes(GetCurrentProcess(), &creation, &exit, &kernel, &user)) {
        return 0.0;
    }
    return fileTimeMilliseconds(kernel) + fileTimeMilliseconds(user);
}

double peakWorkingSetMiB() {
    PROCESS_MEMORY_COUNTERS counters{};
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) {
        return 0.0;
    }
    return static_cast<double>(counters.PeakWorkingSetSize) / (1024.0 * 1024.0);
}
#else
double processCpuMilliseconds() { return 0.0; }
double peakWorkingSetMiB() { return 0.0; }
#endif

CycleTiming runCycle(cat_meme::DnnAnalyzer& analyzer,
                     const cv::Mat& faceFrame, const cv::Rect& face,
                     const cv::Mat& handFrame) {
    const auto cycleStart = Clock::now();
    const auto faceStart = Clock::now();
    const auto faceAnalysis = analyzer.analyze(faceFrame, face);
    const auto faceEnd = Clock::now();
    const auto handAnalysis = analyzer.analyze(handFrame, cv::Rect());
    const auto handEnd = Clock::now();

    double checksum = faceAnalysis.expressionConfidence + handAnalysis.handConfidence;
    checksum += static_cast<double>(faceAnalysis.expressionProbabilities[0]);
    checksum += static_cast<double>(handAnalysis.handLandmarks.size());
    return {milliseconds(faceStart, faceEnd), milliseconds(faceEnd, handEnd),
            milliseconds(cycleStart, handEnd), checksum,
            faceAnalysis.expression != cat_meme::Expression::Unknown,
            handAnalysis.handLandmarks.size()};
}

void printDistribution(const char* prefix, const Distribution& distribution) {
    std::cout << "  \"" << prefix << "_mean_ms\": " << distribution.mean << ",\n"
              << "  \"" << prefix << "_p50_ms\": " << distribution.median << ",\n"
              << "  \"" << prefix << "_p95_ms\": " << distribution.p95 << ",\n"
              << "  \"" << prefix << "_min_ms\": " << distribution.minimum << ",\n"
              << "  \"" << prefix << "_max_ms\": " << distribution.maximum << ",\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        const fs::path source = CAT_MEME_SOURCE_DIR;
        const fs::path models = source / "assets" / "models";

        cv::Mat faceFrame(240, 320, CV_8UC3, cv::Scalar(110, 125, 140));
        cv::circle(faceFrame, cv::Point(160, 120), 70,
                   cv::Scalar(170, 180, 195), -1);
        const cv::Rect face(90, 50, 140, 140);

        const cv::Mat handSource = cv::imread(
            (source / "tests" / "data" /
             "opencv_zoo_gesture_classification.png").string());
        if (handSource.empty() || handSource.cols < 524 || handSource.rows < 196) {
            throw std::runtime_error("The benchmark hand image is missing or malformed");
        }
        const cv::Mat handFrame = handSource(cv::Rect(262, 0, 262, 196)).clone();

        const auto loadStart = Clock::now();
        cat_meme::DnnAnalyzer analyzer(
            models / "facial_expression_recognition_mobilefacenet_2022july.onnx",
            models / "palm_detection_mediapipe_2023feb.onnx",
            models / "handpose_estimation_mediapipe_2023feb.onnx",
            options.useCuda);
        const auto loadEnd = Clock::now();

        if (options.useCuda && !analyzer.usingCuda()) {
            throw std::runtime_error("CUDA was requested but the DNN backend selected CPU");
        }

        const CycleTiming first = runCycle(analyzer, faceFrame, face, handFrame);
        if (!first.expressionProduced || first.handLandmarkCount != 21) {
            throw std::runtime_error(
                "The benchmark inputs did not exercise expression and hand-pose inference");
        }
        if (options.useCuda && !analyzer.usingCuda()) {
            throw std::runtime_error("CUDA inference fell back to CPU during the first cycle");
        }

        for (int i = 0; i < options.warmup; ++i) {
            runCycle(analyzer, faceFrame, face, handFrame);
        }

        std::vector<double> faceTimes;
        std::vector<double> handTimes;
        std::vector<double> cycleTimes;
        faceTimes.reserve(options.iterations);
        handTimes.reserve(options.iterations);
        cycleTimes.reserve(options.iterations);
        double checksum = first.checksum;

        const double cpuStart = processCpuMilliseconds();
        const auto benchmarkStart = Clock::now();
        for (int i = 0; i < options.iterations; ++i) {
            const CycleTiming timing = runCycle(analyzer, faceFrame, face, handFrame);
            faceTimes.push_back(timing.faceMilliseconds);
            handTimes.push_back(timing.handMilliseconds);
            cycleTimes.push_back(timing.totalMilliseconds);
            checksum += timing.checksum;
        }
        const auto benchmarkEnd = Clock::now();
        const double cpuEnd = processCpuMilliseconds();

        if (options.useCuda && !analyzer.usingCuda()) {
            throw std::runtime_error("CUDA inference fell back to CPU during measurement");
        }

        const Distribution faceStats = summarize(faceTimes);
        const Distribution handStats = summarize(handTimes);
        const Distribution cycleStats = summarize(cycleTimes);
        const double wallMilliseconds = milliseconds(benchmarkStart, benchmarkEnd);
        const double cpuMilliseconds = std::max(0.0, cpuEnd - cpuStart);
        const double cyclesPerSecond = 1000.0 * options.iterations / wallMilliseconds;

        std::cout << std::fixed << std::setprecision(3)
                  << "{\n"
                  << "  \"requested_backend\": \""
                  << (options.useCuda ? "CUDA" : "CPU") << "\",\n"
                  << "  \"actual_backend\": \"" << analyzer.backendName() << "\",\n"
                  << "  \"iterations\": " << options.iterations << ",\n"
                  << "  \"warmup_cycles\": " << options.warmup << ",\n"
                  << "  \"opencv_threads\": " << cv::getNumThreads() << ",\n"
                  << "  \"hardware_threads\": " << std::thread::hardware_concurrency() << ",\n"
                  << "  \"model_load_ms\": " << milliseconds(loadStart, loadEnd) << ",\n"
                  << "  \"first_cycle_ms\": " << first.totalMilliseconds << ",\n";
        printDistribution("face", faceStats);
        printDistribution("hand", handStats);
        printDistribution("cycle", cycleStats);
        std::cout << "  \"timed_wall_ms\": " << wallMilliseconds << ",\n"
                  << "  \"process_cpu_ms\": " << cpuMilliseconds << ",\n"
                  << "  \"cpu_core_equivalents\": "
                  << (wallMilliseconds > 0.0 ? cpuMilliseconds / wallMilliseconds : 0.0)
                  << ",\n"
                  << "  \"cycles_per_second\": " << cyclesPerSecond << ",\n"
                  << "  \"images_per_second\": " << cyclesPerSecond * 2.0 << ",\n"
                  << "  \"peak_working_set_mib\": " << peakWorkingSetMiB() << ",\n"
                  << "  \"checksum\": " << checksum << "\n"
                  << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Benchmark error: " << error.what() << '\n';
        return 1;
    }
}
