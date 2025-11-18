#ifndef SCAN_H_
#define SCAN_H_

#include <vector>
#include <iostream>

#define BLOCK_SIZE 512

enum class ScanMethod
{
    KoggeStone,
    Preprint
};

/**
 * @brief Performs scan (prefix sum) on the host (CPU).
 *
 * This function computes the prefix sum of the input data.
 * @param input Input data
 * @return std::vector<int> Output data containing the prefix sums
 */
std::vector<int> scanOnHost(const std::vector<int> &input);

/**
 * @brief Performs scan (prefix sum) on the device (GPU).
 *
 * This function computes the prefix sum of the input data using the specified method.
 * @param input Input data
 * @param method Method to use for scan (default: KoggeStone)
 * @return std::vector<int> Output data containing the prefix sums
 */
std::vector<int> scanOnDevice(const std::vector<int> &input, ScanMethod method = ScanMethod::KoggeStone);

#endif // SCAN_H_
