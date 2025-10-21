#ifndef MATRIX_H_
#define MATRIX_H_

#include <filesystem>
#include <fstream>
#include <vector>

namespace fs = std::filesystem;

/**
 * @brief Simple class for matrix data management
 *
 */
class Matrix
{
public:
    Matrix() = delete;

    /**
     * @brief Construct a new Matrix object
     *
     * @param rows Number of rows
     * @param cols Number of columns
     */
    Matrix(std::size_t rows, std::size_t cols);

    Matrix(const Matrix &) = default;

    Matrix(Matrix &&) noexcept = default;

    Matrix &operator=(const Matrix &) = default;

    Matrix &operator=(Matrix &&) noexcept = default;

    /**
     * @brief Get number of rows
     *
     * @return unsigned int Number of rows
     */
    unsigned int getRows() const;

    /**
     * @brief Get number of columns
     *
     * @return unsigned int Number of columns
     */
    unsigned int getCols() const;

    /**
     * @brief Get const pointer to the matrix data
     *
     * @return const float* Pointer to the matrix data
     */
    const float *getDataConstPtr() const;

    /**
     * @brief Get pointer to the matrix data
     *
     * @return float* Pointer to the matrix data
     */
    float *getDataPtr();

    /**
     * @brief Fill matrix with data
     *
     * @param data Pointer to the data
     * @param size Size of the data
     *
     * @throw std::runtime_error if size does not match matrix dimensions
     */
    void fillData(float *data, unsigned int size);

    /**
     * @brief Write matrix to a file
     *
     * @param filePath Path to the output file
     *
     * @throw std::runtime_error if file cannot be opened
     */
    void writeToFile(fs::path &filePath) const;

private:
    std::size_t rows_;
    std::size_t cols_;
    std::vector<float> data_;
};

/**
 * @brief Read matrix from a file
 *
 * @param filePath Path to the input file
 * @return Matrix Read matrix
 *
 * @throw std::runtime_error if file cannot be opened
 */
Matrix readFromFile(fs::path &filePath);

/**
 * @brief Generate a random matrix
 *
 * @param rows Number of rows
 * @param cols Number of columns
 * @return Matrix Generated random matrix
 */
Matrix generateRandomMatrix(unsigned int rows, unsigned int cols);

#endif /* MATRIX_H_ */