#include <iostream>
#include <filesystem>
#include <cmath>
#include "image.h"

namespace fs = std::filesystem;

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <image1> <image2>" << std::endl;
        return 1;
    }

    fs::path imgFile1 = argv[1];
    fs::path imgFile2 = argv[2];

    if (!fs::exists(imgFile1))
    {
        std::cerr << "Input file does not exist: " << imgFile1 << std::endl;
        return 1;
    }

    if (!fs::exists(imgFile2))
    {
        std::cerr << "Input file does not exist: " << imgFile2 << std::endl;
        return 1;
    }

    std::cout << "Reading input images..." << std::endl;
    Image img1 = readImageFromPPM(imgFile1);
    Image img2 = readImageFromPPM(imgFile2);

    if (img1.getWidth() != img2.getWidth() || img1.getHeight() != img2.getHeight())
    {
        std::cerr << "Input images must have the same dimensions." << std::endl;
        return 1;
    }

    if (img1.isGray() != img2.isGray())
    {
        std::cerr << "Input images must both be grayscale or both be RGB." << std::endl;
        return 1;
    }

    std::cout << "Computing image difference..." << std::endl;
    Image diffImg(img1.getWidth(), img1.getHeight(), img1.isGray());
    Image diffMap(img1.getWidth(), img1.getHeight(), true); // Always grayscale for diff map
    const float *data1 = img1.getDataConstPtr();
    const float *data2 = img2.getDataConstPtr();
    float *diffData = diffImg.getDataPtr();
    float *mapData = diffMap.getDataPtr();
    bool sameImages = true;

    for (std::size_t i = 0; i < img1.getRows() * img1.getCols(); ++i)
    {
        diffData[i] = std::fabs(data1[i] - data2[i]);
        if (diffData[i] > 1e-6) // Threshold for floating point comparison
        {
            sameImages = false;
            mapData[i / (img1.isGray() ? 1 : 3)] = 255.0f; // Mark difference in the map
        }
    }

    if (sameImages)
    {
        std::cout << "\033[32mThe images are identical.\033[0m" << std::endl;
    }
    else
    {
        std::cout << "\033[31mThe images differ.\033[0m" << std::endl;

        fs::path outFile = "diff_img.ppm";
        diffImg.writeToPPM(outFile);
        std::cout << "Difference image written to " << outFile << std::endl;

        fs::path outFileMap = "diff_img_map.ppm";
        diffMap.writeToPPM(outFileMap);
        std::cout << "Difference map written to " << outFileMap << std::endl;
    }
    return 0;
}