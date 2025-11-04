#include "conv_2d.h"

__constant__ float c_mask[MAX_MASK_SIZE * MAX_MASK_SIZE];

__global__ void basicConvolution2D(float *output, const float *input, const int width, const int height, const int maskSize)
{
}

__global__ void tiledConvolution2D(float *output, const float *input, const int width, const int height, const int maskSize)
{
}

Image convolutionOnDevice(const Image &img, const std::vector<float> &mask, ConvMethod method)
{
}

Image convolutionOnHost(const Image &img, const std::vector<float> &mask)
{
    // Get image dimensions
    int height = img.getHeight();
    int width = img.getWidth();
    int maskSize = static_cast<int>(std::sqrt(mask.size()));
    int r = maskSize / 2;

    // Create output image
    Image output(width, height, img.isGray());

    // Perform convolution
    for (int row = 0; row < height; ++row)
    {
        for (int col = 0; col < width; ++col)
        {
            float pixelValue = 0.0f;
            for (int mRow = -r; mRow <= r; ++mRow)
            {
                for (int mCol = -r; mCol <= r; ++mCol)
                {
                    int imgRow = row + mRow;
                    int imgCol = col + mCol;
                    float imgValue = 0.0f;

                    if (imgRow >= 0 && imgRow < height && imgCol >= 0 && imgCol < width)
                    {
                        imgValue = img.getDataConstPtr()[imgRow * width + imgCol];
                    }

                    float maskValue = mask[(mRow + r) * maskSize + (mCol + r)];
                    pixelValue += imgValue * maskValue;
                }
            }
            output.getDataPtr()[row * width + col] = pixelValue;
        }
    }

    return output;
}
