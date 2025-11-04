#include "image.h"

Image::Image(std::size_t width, std::size_t height, bool isGray) : Matrix(height, isGray ? width : width * 3), isGray_(isGray), width_(width), height_(height)
{
}

std::size_t Image::getWidth() const
{
    return width_;
}

std::size_t Image::getHeight() const
{
    return height_;
}

bool Image::isGray() const
{
    return isGray_;
}

void Image::writeToPPM(const fs::path &filePath) const
{
    writePPM(filePath.string().c_str(), getDataConstPtr(), width_, height_, isGray_);
}

Image readImageFromPPM(const fs::path &filePath)
{
    unsigned int width, height;
    getPPMSize(filePath.string().c_str(), &width, &height);

    // NOTE: PPM files are always RGB, so we read 3 channels.
    // If the image is grayscale, the same value will be stored in R, G, and B channels.
    // Therefore, after reading, we will get rid of the extra channels if needed.
    std::vector<float> tmpData(width * height * 3);
    readPPM(filePath.string().c_str(), tmpData.data());

    // Determine if the image is grayscale by checking if all R, G, B values are the same for each pixel
    bool isGray = true;
    for (std::size_t i = 0; i < width * height; ++i)
    {
        if (!(tmpData[3 * i] == tmpData[3 * i + 1] && tmpData[3 * i] == tmpData[3 * i + 2]))
        {
            isGray = false;
            break;
        }
    }

    // If the image is grayscale, we only need one channel
    std::vector<float> outputData(width * height * (isGray ? 1 : 3));
    if (isGray)
    {
        for (std::size_t i = 0; i < width * height; ++i)
        {
            outputData[i] = tmpData[3 * i];
        }
    }
    else
    {
        outputData = std::move(tmpData);
    }

    Image img(width, height, isGray);
    img.fillData(outputData.data(), outputData.size());

    return img;
}