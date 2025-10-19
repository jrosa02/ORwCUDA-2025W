#ifndef BLUR_H_
#define BLUR_H_

#include "image.h"

/**
 * @brief Apply a box blur to the input image using a square kernel of given size
 *
 * This function uses standard CPU computation to apply a box blur to the input image.
 * @param input Input image
 * @param kernelSize Size of the square kernel (must be odd)
 * @return Image Blurred image
 */
Image imageBlurOnHost(const Image &input, int kernelSize);

/**
 * @brief Apply a box blur to the input image using a square kernel of given size
 *
 * This function uses GPU parallel computing to apply a box blur to the input image.
 * @param input Input image
 * @param kernelSize Size of the square kernel (must be odd)
 * @return Image Blurred image
 */
Image imageBlurOnDevice(const Image &input, int kernelSize);

#endif /* BLUR_H_ */