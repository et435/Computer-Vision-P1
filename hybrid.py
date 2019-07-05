#Eshann Toteja
#NetID: et435

import sys
sys.path.append('/Users/kb/bin/opencv-3.1.0/build/lib/')

import cv2
import numpy as np



def cross_correlation_2d(img, kernel):

    imageRow = img.shape[0]
    imageCol = img.shape[1]

    if len(img.shape) == 3: #RGB
        imageR = np.zeros([imageRow, imageCol])
        imageG = np.zeros([imageRow, imageCol])
        imageB = np.zeros([imageRow, imageCol])

        imageR = img[:, :, 0]
        imageG = img[:, :, 1]
        imageB = img[:, :, 2]

        m, n = kernel.shape
        xPad = (m - 1) / 2
        yPad = (n - 1) / 2

        paddedR = np.pad(imageR, ((xPad, xPad), (yPad, yPad)), 'constant')
        paddedG = np.pad(imageG, ((xPad, xPad), (yPad, yPad)), 'constant')
        paddedB = np.pad(imageB, ((xPad, xPad), (yPad, yPad)), 'constant')

        imageRCC = np.zeros([imageRow, imageCol])
        imageGCC = np.zeros([imageRow, imageCol])
        imageBCC = np.zeros([imageRow, imageCol])

        for x in xrange(imageRow):
            for y in xrange(imageCol):
                imageRCC[x, y] = (paddedR[x:x + m, y:y + n] * kernel).sum()
                imageGCC[x, y] = (paddedG[x:x + m, y:y + n] * kernel).sum()
                imageBCC[x, y] = (paddedB[x:x + m, y:y + n] * kernel).sum()

        imageRCC = np.reshape(imageRCC, (imageRow, imageCol, 1))
        imageGCC = np.reshape(imageGCC, (imageRow, imageCol, 1))
        imageBCC = np.reshape(imageBCC, (imageRow, imageCol, 1))

        imageComplete = np.concatenate((imageRCC, imageGCC, imageBCC), axis=2)


    else:   #Grayscale
        m, n = kernel.shape
        xPad = (m-1) / 2
        yPad = (n-1) / 2

        padded = np.pad(img, ((xPad, xPad), (yPad, yPad)), 'constant')

        imageComplete = np.zeros([imageRow, imageCol])

        for x in xrange(imageRow):
            for y in xrange(imageCol):
                imageComplete[x, y] = (padded[x:x + m, y:y + n] * kernel).sum()

    return imageComplete



def convolve_2d(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    temp = np.flip(kernel, axis=1) #Flip over x axis
    flippedKernel = np.flip(temp, axis=0)   #Flip over y axis
    return cross_correlation_2d(img, flippedKernel)

def gaussian_blur_kernel_2d(sigma, height, width):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions height x width such that convolving it
        with an image results in a Gaussian-blurred image.
    '''
    gaussianKernel = np.zeros([height, width])

    m = width / 2
    n = height / 2

    for x in xrange(height):
        for y in xrange(width):
            gaussianKernel[x,y] = gaussianCalculation(x-n,y-m,sigma)

    answer = 1/gaussianKernel.sum() * gaussianKernel

    return answer

def gaussianCalculation(x, y, sigma):
    exp = -(float(x**2 + y**2) / (2 * (sigma**2)))
    return (1 / ((2 * np.pi) * (sigma**2))) * np.power(np.e,exp)



def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    kernelLP = gaussian_blur_kernel_2d(sigma,size,size)
    return convolve_2d(img,kernelLP)

def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    scaledImpulse = np.zeros([size,size])
    scaledImpulse[size/2][size/2] = 1
    kernelHP = gaussian_blur_kernel_2d(sigma, size, size)
    kernel = scaledImpulse - kernelHP
    return convolve_2d(img,kernel)


def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *= 2 * (1 - mixin_ratio)
    img2 *= 2 * mixin_ratio
    hybrid_img = (img1 + img2)
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)


