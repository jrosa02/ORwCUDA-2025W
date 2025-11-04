#ifndef CONV_1D_H_
#define CONV_1D_H_

#include <vector>

constexpr int TILE_SIZE = 256;
constexpr int MAX_MASK_WIDTH = 21;

enum class ConvMethod
{
    Basic,
    Tiled,
};

/**
 * @brief Performs 1D convolution on the host (CPU).
 *
 * This function performs a 1D convolution of the input signal with the given mask.
 * @param signal Input signal
 * @param mask Convolution mask
 * @return std::vector<float> Resulting convolved signal
 */
std::vector<float> convolutionOnHost(const std::vector<float> &signal, const std::vector<float> &mask);

/**
 * @brief Performs 1D convolution on the device (GPU).
 *
 * This function performs a 1D convolution of the input signal with the given mask using the specified method.
 * @param signal Input signal
 * @param mask Convolution mask
 * @param method Method to use for convolution (default: Basic)
 * @return std::vector<float> Resulting convolved signal
 */
std::vector<float> convolutionOnDevice(const std::vector<float> &signal, const std::vector<float> &mask, ConvMethod method = ConvMethod::Basic);

#endif // CONV_1D_H_