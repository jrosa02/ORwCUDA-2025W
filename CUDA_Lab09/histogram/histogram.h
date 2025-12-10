#ifndef HISTOGRAM_H_
#define HISTOGRAM_H_

#include <stdexcept>
#include <vector>

constexpr int N_LETTERS = 26;

enum class HistMethod
{
    Block,
    Interleaved,
    Privatised,
};

/**
 * @brief Computes the histogram of the input data using the specified method.
 *
 * This function computes the histogram of the input data buffer using the GPU.
 * @param data Input data buffer
 * @param nBins Number of histogram bins
 * @param method Method to use for histogram computation (default: Basic)
 * @return std::vector<unsigned int> Computed histogram
 */
std::vector<unsigned int> computeHistogramOnDevice(const std::vector<unsigned char> &data, int nBins, HistMethod method = HistMethod::Block);

/**
 * @brief Computes the histogram of the input data on the host (CPU).
 *
 * This function computes the histogram of the input data buffer using standard CPU computation.
 * @param data Input data buffer
 * @param nBins Number of histogram bins
 * @return std::vector<unsigned int> Computed histogram
 */
std::vector<unsigned int> computeHistogramOnHost(const std::vector<unsigned char> &data, int nBins);

#endif /* HISTOGRAM_H_ */