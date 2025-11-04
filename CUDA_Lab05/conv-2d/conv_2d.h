#ifndef CONV_2D_H_
#define CONV_2D_H_

#include "image.h"

constexpr int TILE_SIZE = 16;
constexpr int MAX_MASK_SIZE = 15;

enum class ConvMethod
{
    Basic,
    Tiled,
};

/**
 * @brief Performs 2D convolution on the host (CPU).
 *
 * This function performs a 2D convolution of the input image with the given mask.
 * @param img Input image
 * @param mask Convolution mask
 * @return Image Resulting convolved image
 */
Image convolutionOnHost(const Image &img, const std::vector<float> &mask);

/**
 * @brief Performs 2D convolution on the device (GPU).
 *
 * This function performs a 2D convolution of the input image with the given mask using the specified method.
 * @param image Input image
 * @param mask Convolution mask
 * @param method Method to use for convolution (default: Basic)
 * @return Image Resulting convolved image
 */
Image convolutionOnDevice(const Image &img, const std::vector<float> &mask, ConvMethod method = ConvMethod::Basic);

#endif // CONV_2D_H_