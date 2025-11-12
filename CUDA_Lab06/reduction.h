#ifndef REDUCTION_H_
#define REDUCTION_H_

#include <vector>

#define BLOCK_SIZE 256

enum class ReductionMethod
{
    Basic,
    Optimized,
    CooperativeGroups
};

/**
 * @brief Performs reduction on the host (CPU).
 *
 * This function computes the sum of all elements in the input data.
 * @param data Input data
 * @return int Sum of all elements
 */
int reductionOnHost(const std::vector<int> &data);

/**
 * @brief Performs reduction on the device (GPU).
 *
 * This function computes the sum of all elements in the input data using the specified method.
 * @param data Input data
 * @param method Method to use for reduction (default: Basic)
 * @return int Sum of all elements
 */
int reductionOnDevice(const std::vector<int> &data, ReductionMethod method = ReductionMethod::Basic);

#endif // REDUCTION_H_
