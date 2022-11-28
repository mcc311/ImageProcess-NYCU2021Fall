import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from numba import njit
from collections import Counter
import os


class PixelOp:
    """
    A class obtains two classes containing pixel operations methods.

    ...

    sub_classes
    -----------
    GrayLevelTransform
    ContrastStretching

    """

    class GrayLevelTransform:
        """
        A class obtains three gray level transform methods.

        ...

        Methods
        -------
        static(negative)(image:np.array) -> np.array
            Calculate output image by L - image
        static(power_low)(image:np.array) -> np.array
            Calculate output image by $s=c * r^\gamma$.
        static(log)(image:np.array) -> np.array
            Calculate output image by $s = c *log(r + 1)$.
        """

        @staticmethod
        def negative(image) -> np.array:
            """
            Calculate output image by L - image

            :param image: np.array
            :return: np.array
            """
            return 255 - image

        @staticmethod
        def power_law(image: np.array, c: float = 1., gamma: float = 1.) -> np.array:
            """
            Calculate output image by $s=c * r^\gamma$.

            :param image: np.array
                Input image.
            :param c: float
                Constant represents adjust size.
            :param gamma: float
                Constant.Sometimes indicates different display monitor.
            :return: np.array
                Output Image.
            """
            return np.array((image / 255) ** gamma * c * 255, dtype='uint8')

        @staticmethod
        def log(image: np.array) -> np.array:
            """
            Calculate output image by $s = c *log(r + 1)$.

            :param image: np.array
            :return: np.array
            """
            c = 255 / (np.log2(1 + 255))
            log_transformed = np.log2(image + 1) * c
            return np.array(log_transformed)

        @staticmethod
        def threshold(image, lower, upper, binary=True, **kwargs):
            if not kwargs['invert']:
                result = (image >= lower) * (image <= upper)
            else:
                result = np.invert((image >= lower) * (image <= upper))
            if not binary:
                result = image[result]
            return result

    class ContrastStretching:
        """
        A class obtains contrast stretching methods

        ...

        Methods
        -------
        static(percentile)(image, r1, r2, s1=0, s2=255)
            Calculate output image by linear function with two break points.

        static(hist_eq)(image)
            Calculate output image with histogram equalization algorithms.
        """

        @staticmethod
        def percentile(image: np.array, r1: float, r2: float, s1: float = 0, s2=255) -> np.array:
            """
            Calculate output image by linear function with two breaking points.
            :param image: np.array
            :param r1:  np.array
                represent the x value of first breaking point.
            :param r2: np.array
                represent the x value of second breaking point.
            :param s1: np.array
                represent the y value of first breaking point.
            :param s2: np.array
                represent the y value of second breaking point.
            :return: np.array
                Output image.
            """

            result = image.copy().astype('float32')
            result[image <= r1] *= s1 / r1
            result[np.multiply(image > r1, image < r2)
                   ] *= (s2 - s1) / (r2 - r1)
            result[np.multiply(image > r1, image < r2)
                   ] += (r2 * s1 - r1 * s2) / (r2 - r1)
            result[image >= r2] *= (256 - s2) / (256 - r2)
            result[image >= r2] += 256 * (s2 - r2) / (256 - r2)
            return result

        @staticmethod
        def hist_eq(image):
            """
            Calculate output image with histogram equalization algorithms.
            ref: Acharya and Ray, Image Processing: Principles and Applications, Wiley-Interscience 2005 ISBN 0471719986
            :param image: np.array
                Input image.
            :return: np.array
                Output image.
            """

            def histogram(array):
                hist = {pixel: 0 for pixel in range(256)}
                hist.update(Counter(array))
                return np.array(sorted(hist.items(), key=lambda x: x[0]))[:, 1]

            hist = histogram(image.ravel())
            pdf = hist / image.size
            cdf = pdf.cumsum()
            equ_value = np.around(cdf * 255).astype('uint8')
            result = equ_value[image]
            return result


class NeighborhoodOp:
    """
    A class obtains neighborhood operation methods.

    ...

    Methods
    -------
    static(padding)(image, k_size, padding_mode='replicate')
        Remaking the outer-frame of image makes the convolution easier.

    static(njit(convolution))(image, kernel, padding_image)
        Do the discrete convolution.

    static(njit(median_filter))(image, m, n, padding_image)
        Return the output image calculated by convolution with median kernel.

    static(filtering)(image, k_size, mode, **kwargs)
        Return the output image calcilated by convolution with kernel depending on *mode* args.

    """
    @staticmethod
    def padding(image, left, right, top, bottom, padding_mode='replicate'):
        """
        Remaking the outer-frame of image makes the convolution easier.
        :param image: np.array
            Input image.
        :param k_size: int
            Kernel size. It should be odd.
        :param padding_mode: str
            There are two different mode doing padding.
            zero:
                Fill the outer-frame with zeros.
            replicate:
                Fill the outer-frame with neighbor pixels.
        :return: np.array
            Output image with outer-frame.
        """

        padding_image = np.zeros(
            np.array(image.shape) + np.array((top+bottom, right+left)))
        top = top if top else None
        bottom = -bottom if bottom else None
        left = left if left else None
        right = -right if right else None
        padding_image[top: bottom, left:right] = image
        if padding_mode == 'zero':
            return padding_image
        if padding_mode == 'replicate':
            if top:
                for i in range(top):
                    padding_image[i] = padding_image[top]
            if bottom:
                for i in range(-bottom):
                    padding_image[-(i+1)] = padding_image[bottom-1]
            if left:
                for j in range(left):
                    padding_image[:, j] = padding_image[left]
            if right:
                for j in range(-right):
                    padding_image[:, -(j+1)] = padding_image[right-1]
        return padding_image

    @staticmethod
    @njit
    def convolution(image, kernel, padding_image, mode='conv2d'):
        """
        Do the discrete convolution.
        The numba.njit decorator is used to accelerate the matrix computation.
        :param image: np.array
            Input image.
        :param kernel: np.array
            The kernel(filter/mask) using in convolution.
            kernel.shape[i] is odd, for any legal i.
        :param padding_image:  np.array
            The input image with outer-frame. It's the base image during the convolution.
        :return: np.array
            Output image.
        """

        m, n = kernel.shape
        new_image = np.zeros(image.shape)
        img_m, img_n = image.shape
        for i in range(img_m):
            for j in range(img_n):
                if mode == 'conv2d':
                    new_image[i, j] = np.sum(
                        kernel * padding_image[i:i + m, j:j + n])
                if mode == 'erosion':
                    new_image[i, j] = (
                        kernel * padding_image[i:i + m, j:j + n]).all() * 255
                if mode == 'dilation':
                    new_image[i, j] = (
                        kernel * padding_image[i:i + m, j:j + n]).any() * 255
                if mode == 'custom1':
                    new_image[i, j] = (
                        not (kernel * padding_image[i:i + m, j:j + n]).all()) * image[i, j] * 255
        return new_image

    @staticmethod
    @njit
    def median_filter(image, m, n, padding_image):
        """
        Return the output image calculated by convolution with median kernel.
        It is often used to remove the noise of image, such like "salt and pepper".
        :param image: np.array
            Input image.
        :param m: int
            Columns size of kernel. It should be odd.
        :param n: int
            Row size of kernel. It should be odd.
        :param padding_image: np.array
            The input image with outer-frame. It's the base image during the convolution.

        :return: np.array
            Output image.
        """
        new_image = np.zeros(image.shape)

        img_m, img_n = image.shape
        for i in range(img_m):
            for j in range(img_n):
                new_image[i, j] = np.median(padding_image[i:i + m, j:j + n])
        return new_image

    @staticmethod
    def filtering(image, k_size=3, mode='median', **kwargs):
        """
        Return the output image calcilated by convolution with kernel depending on *mode* args.
        :param image: np.array
            Input image.
        :param k_size: int
            Kernel size. It should be odd.
        :param mode: str
            Represent which method the input image should be calculated with.
            median:
                Calculate the median of every k_size*k_size pixel around the input as the output.

            laplacian:
                Use second derivation of neighbors(without corners) as the kernel.

            laplacian-corner:
                Use second derivation of neighbors(including corners) as the kernel.

            sharperning:
                Use second derivation of neighbors(without corners) plus 1 in the center as the kernel.

            sharpening_corner:
                Use second derivation of neighbors(including corners) plus 1 in the center as the kernel.

            average:
                Use the average mask which will calculate the average value of neighbor pixels.
            gaussian:
                Use Gaussian filter as the kernel.
        :param kwargs:
            padding_mode: str
                The input image with outer-frame. It's the base image during the convolution.
        :return: np.array
        Output image.
        """
        m = k_size
        n = k_size
        if mode == 'median':
            padding_image = NeighborhoodOp.padding(image, k_size, **kwargs)
            return NeighborhoodOp.median_filter(image, m, n, padding_image)
        else:
            kernel = None
            if mode == 'laplacian':
                kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

            if mode == 'laplacian-corner':
                kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

            if mode == 'sharpening':
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

            if mode == 'sharpening_corner':
                kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            if mode == 'average':
                kernel = np.ones((k_size, k_size)) / (k_size * k_size)

            if mode == 'gaussian':
                x, y = np.mgrid[-(k_size - 1) // 2:(k_size - 1) //
                                2 + 1, -(k_size - 1) // 2:(k_size - 1) // 2 + 1]
                kernel = np.exp(-(x ** 2 + y ** 2))
                kernel = kernel / kernel.sum()

            if kernel is None:
                raise ValueError('Name of args: mode, is wrong!')
            if len(image.shape) == 3:
                result = image.copy()
                for channel in range(image.shape[-1]):
                    padding_image = NeighborhoodOp.padding(
                        image[:, :, channel], k_size, **kwargs)
                    result[:, :, channel] = NeighborhoodOp.convolution(
                        image[:, :, channel], kernel, padding_image)
            else:
                padding_image = NeighborhoodOp.padding(image, k_size, **kwargs)
                result = NeighborhoodOp.convolution(
                    image, kernel, padding_image)
            return result

    @staticmethod
    def morphologyEx(image, kernel=None, mode='erosion', iteration=1, **kwargs):
        if kernel is None:
            kernel = np.ones((3, 3))
        if 'padding' not in kwargs:
            padding = kernel.shape[0]
        else:
            padding = kwargs['padding']
        for _ in range(iteration):
            padding_image = NeighborhoodOp.padding(image, padding)
            image = NeighborhoodOp.convolution(
                image, kernel, padding_image, mode=mode)
        return image


if __name__ == '__main__':
    if not os.path.exists('output'):
        os.mkdir('output')

    # Negative Transform
    img = Image.open('input/xray.jpg')
    img = np.array(img)
    negative = PixelOp.GrayLevelTransform.negative(img)
    Image.fromarray(negative).convert('L').save('output/negative.jpg')

    # Log Transform
    img = np.array(Image.open('input/night.jpg'), dtype='int')
    log = img.copy()
    for channel in range(3):
        log[:, :, channel] = PixelOp.GrayLevelTransform.log(img[:, :, channel])
    log = log.astype('uint8')
    Image.fromarray(log).convert('RGB').save('output/log.jpg')

    # Power-Law Transform
    for gamma in [.3, .5, .8, 1.2, 1.5, 2]:
        power_law = PixelOp.GrayLevelTransform.power_law(img, gamma=gamma)
        Image.fromarray(power_law).save(f'output/PowerLaw_{gamma}.jpg')

    # Contrast stretching
    img = Image.open('input/low_contrast.jpg').convert('L')
    img = np.array(img)

    # min-Max
    minmax = PixelOp.ContrastStretching.percentile(
        img, r1=np.min(img), s1=0, r2=np.max(img), s2=255)
    Image.fromarray(minmax).convert('L').save('output/minmax.jpg')

    # min-Max with c
    c = 15
    minmax_c = PixelOp.ContrastStretching.percentile(
        img, r1=np.min(img) + c, s1=0, r2=np.max(img) - c, s2=255)
    Image.fromarray(minmax_c).convert('L').save('output/minmax_c.jpg')

    # Histogram Equalization
    he = PixelOp.ContrastStretching.hist_eq(img)
    Image.fromarray(he).convert('L').save('output/he.jpg')

    # Compare CDF
    fig = plt.figure(figsize=(6, 4))
    cum = 1
    plt.hist(img.ravel(), bins=256, density=1,
             cumulative=cum, label='origin', histtype='step')
    plt.hist(minmax.ravel(), bins=256, density=1,
             cumulative=cum, label='min-Max', histtype='step')
    plt.hist(minmax_c.ravel(), bins=256, density=1, cumulative=cum,
             label=f'min-Max with c={c}', histtype='step')
    plt.hist(he.ravel(), bins=256, density=1,
             cumulative=cum, label='HE', histtype='step')
    plt.legend(loc='upper left')
    plt.savefig('output/contrast_hist.png')

    # Neighborhood Op
    img = Image.open('input/pepper.jpeg').convert('L')
    img = np.array(img)

    # Sharpening
    sharpen = NeighborhoodOp.filtering(img, mode='sharpening')
    sharpen_corner = NeighborhoodOp.filtering(img, mode='sharpening_corner')
    Image.fromarray(sharpen).convert('L').save('output/sharpening.jpg')
    Image.fromarray(sharpen_corner).convert(
        'L').save('output/sharpening_corner.jpg')

    # Median Filter
    med1 = NeighborhoodOp.filtering(img, mode='median')
    Image.fromarray(med1).convert('L').save('output/median1.jpg')
    med2 = NeighborhoodOp.filtering(med1, mode='median')
    Image.fromarray(med2).convert('L').save('output/median2.jpg')

    # Smoothing
    img = Image.open('output/sharpening_corner.jpg')
    img = np.array(img)
    average = NeighborhoodOp.filtering(img, mode='average')
    Image.fromarray(average).convert('L').save('output/average.jpg')

    gaussian = NeighborhoodOp.filtering(img, mode='gaussian')
    Image.fromarray(gaussian).convert('L').save('output/gaussian.jpg')

    # Edge Detection
    img = Image.open('output/median2.jpg')
    img = np.array(img)
    edge = NeighborhoodOp.filtering(img, mode='laplacian')
    Image.fromarray(edge).convert('L').save('output/edge.jpg')
    edge_corner = NeighborhoodOp.filtering(img, mode='laplacian-corner')
    Image.fromarray(edge_corner).convert('L').save('output/edge_corner.jpg')

    # Minus Smooth
    img = Image.open('input/night.jpg')
    img = np.array(img)

    edge_night = NeighborhoodOp.filtering(img, mode='laplacian')
    Image.fromarray(edge_night).convert('L').save('output/edge_night.jpg')
    edge_night_corner = NeighborhoodOp.filtering(img, mode='laplacian-corner')
    Image.fromarray(edge_night_corner).convert(
        'L').save('output/edge_night_corner.jpg')

    edge_gaussian = img - NeighborhoodOp.filtering(img, mode='gaussian')
    Image.fromarray(edge_gaussian).convert(
        'L').save('output/edge_gaussian.jpg')
    edge_average = img - NeighborhoodOp.filtering(img, mode='average')
    Image.fromarray(edge_average).convert('L').save('output/edge_average.jpg')
