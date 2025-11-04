#ifndef IMAGE_H_
#define IMAGE_H_

#include <filesystem>

#include "matrix.h"
#include "ppmIO.h"

namespace fs = std::filesystem;

/**
 * @brief Simple class for image data management. Inherits from Matrix for data storage.
 *
 */
class Image : public Matrix
{
public:
    /**
     * @brief Construct a new Image object
     *
     * @param width Image width
     * @param height Image height
     * @param isGray True if image is grayscale, false if RGB
     */
    Image(std::size_t width, std::size_t height, bool isGray = false);

    bool isGray() const;
    std::size_t getWidth() const;
    std::size_t getHeight() const;

    /**
     * @brief Write image to a PPM file
     *
     * @param filePath Path to the output file
     */
    void writeToPPM(const fs::path &filePath) const;

private:
    bool isGray_;
    std::size_t width_;
    std::size_t height_;
};

/**
 * @brief Read an image from a PPM file
 *
 * @param filePath Path to the input PPM file
 * @return Image Returned image
 */
Image readImageFromPPM(const fs::path &filePath);

#endif /* IMAGE_H_ */