import numpy
import cv2
import pywt


def cvt_wavelet(img, mode='haar', level=1):
    img_array = img
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    img_array = numpy.float32(img_array)
    # Now we need values in the range 0 and 1 for wavelet transform:
    img_array /= 255
    # db1 <=> haar wavelet
    coefficients = pywt.wavedec2(img_array, mode, level=level)

    coefficients_h = list(coefficients)
    coefficients_h[0] *= 0

    img_array_h = pywt.waverec2(coefficients_h, mode)
    img_array_h *= 255
    img_array_h = numpy.uint8(img_array_h)

    return img_array_h
