#pragma once

namespace cat_meme {

[[nodiscard]] bool shouldDisplayMeme(bool faceDetected, bool hasMatch,
                                     double similarity, double threshold);

}  // namespace cat_meme
