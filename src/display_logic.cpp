#include "cat_meme/display_logic.hpp"

namespace cat_meme {

bool shouldDisplayMeme(bool faceDetected, bool hasMatch,
                       double similarity, double threshold) {
    return hasMatch && (faceDetected || similarity >= threshold);
}

}  // namespace cat_meme
