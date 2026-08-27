cmake_minimum_required(VERSION 3.20)

project(cat_meme_detector VERSION 1.0.0 LANGUAGES CXX)

option(CAT_MEME_BUILD_TESTS "Build the regression tests" ON)

# OpenCV libraries are static, while all targets use the DLL MSVC runtime. This
# also lets OpenCV's optional ASM DNN kernels build with current CMake versions.
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

find_package(OpenCV 5.0 REQUIRED COMPONENTS
    core
    dnn
    features
    geometry
    highgui
    imgcodecs
    imgproc
    videoio
    xobjdetect
)

add_library(cat_meme_core
    src/display_logic.cpp
    src/face_detector.cpp
    src/feature_extractor.cpp
    src/hand_gesture_stabilizer.cpp
    src/image_utils.cpp
    src/meme_database.cpp
    src/semantic_analysis.cpp
)

target_include_directories(cat_meme_core
    PUBLIC include
    PRIVATE ${OpenCV_INCLUDE_DIRS}
)
target_compile_features(cat_meme_core PUBLIC cxx_std_17)
target_link_libraries(cat_meme_core PUBLIC ${OpenCV_LIBS})

if(MSVC)
    target_compile_options(cat_meme_core PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(cat_meme_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(cat_meme_detector src/main.cpp)
target_link_libraries(cat_meme_detector PRIVATE cat_meme_core)
target_compile_definitions(cat_meme_detector PRIVATE
    CAT_MEME_SOURCE_ASSETS="${CMAKE_CURRENT_SOURCE_DIR}/assets"
)

add_custom_target(cat_meme_assets ALL
    COMMAND ${CMAKE_COMMAND} -E remove_directory
            "${CMAKE_CURRENT_BINARY_DIR}/assets.staging"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/assets"
            "${CMAKE_CURRENT_BINARY_DIR}/assets.staging"
    COMMAND ${CMAKE_COMMAND} -E remove_directory
            "${CMAKE_CURRENT_BINARY_DIR}/assets"
    COMMAND ${CMAKE_COMMAND} -E rename
            "${CMAKE_CURRENT_BINARY_DIR}/assets.staging"
            "${CMAKE_CURRENT_BINARY_DIR}/assets"
    COMMENT "Copying authored meme and detector assets"
)
add_dependencies(cat_meme_detector cat_meme_assets)

if(CAT_MEME_BUILD_TESTS)
    include(CTest)
    add_executable(cat_meme_tests tests/tests.cpp)
    target_link_libraries(cat_meme_tests PRIVATE cat_meme_core)
    target_compile_definitions(cat_meme_tests PRIVATE
        CAT_MEME_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
        CAT_MEME_BINARY_DIR="${CMAKE_CURRENT_BINARY_DIR}"
    )
    add_test(NAME cat_meme_tests COMMAND cat_meme_tests)
    add_test(NAME cat_meme_bundled_assets
             COMMAND cat_meme_detector --check-assets)
endif()
