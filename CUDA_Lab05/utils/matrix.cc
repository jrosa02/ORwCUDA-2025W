#include "matrix.h"

Matrix::Matrix(std::size_t rows, std::size_t cols) : rows_(rows), cols_(cols), data_(rows * cols)
{
    if (rows == 0 || cols == 0)
    {
        throw std::runtime_error("Matrix dimensions must be greater than zero");
    }
}

unsigned int Matrix::getRows() const
{
    return rows_;
}

unsigned int Matrix::getCols() const
{
    return cols_;
}

const float *Matrix::getDataConstPtr() const
{
    return data_.data();
}

float *Matrix::getDataPtr()
{
    return data_.data();
}

void Matrix::fillData(float *data, unsigned int size)
{
    if (size != rows_ * cols_)
    {
        throw std::runtime_error("Data size does not match matrix size");
    }
    std::copy(data, data + size, data_.begin());
}

void Matrix::writeToFile(fs::path &filePath) const
{
    std::ofstream file(filePath);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file: " + filePath.string());
    }

    // Write matrix size
    file << rows_ << " " << cols_ << std::endl;

    // Write matrix data
    for (std::size_t i = 0; i < rows_; ++i)
    {
        for (std::size_t j = 0; j < cols_; ++j)
        {
            file << data_[i * cols_ + j] << " ";
        }
        file << std::endl;
    }

    file.close();
}

Matrix readFromFile(fs::path &filePath)
{
    std::ifstream file(filePath);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file: " + filePath.string());
    }

    // Create matrix
    unsigned int rows, cols;
    file >> rows >> cols;
    Matrix mat(rows, cols);

    // Read matrix data
    float *data = new float[rows * cols];
    for (unsigned int i = 0; i < rows * cols; ++i)
    {
        file >> data[i];
    }
    file.close();

    // Fill matrix with data
    mat.fillData(data, rows * cols);
    delete[] data;

    return mat;
}

Matrix generateRandomMatrix(unsigned int rows, unsigned int cols)
{
    Matrix mat(rows, cols);
    float *data = new float[rows * cols];
    for (unsigned int i = 0; i < rows * cols; ++i)
    {
        data[i] = static_cast<float>((std::rand() % 20)); // Random float between 0 and 20
    }
    mat.fillData(data, rows * cols);
    delete[] data;
    return mat;
}