# coding=utf-8
__author__ = "Dimitrios Karkalousos"

# Parts of the code have been taken from https://github.com/facebookresearch/fastMRI

import numpy as np
from runstats import Statistics
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def mse(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Mean Squared Error (MSE).

    Parameters
    ----------
    x : np.ndarray
        Target image. It must be a 3D array, where the first dimension is the number of slices. In case of 2D images,
        the first dimension should be 1.
    y : np.ndarray
        Predicted image. It must be a 3D array, where the first dimension is the number of slices. In case of 2D
        images, the first dimension should be 1.

    Returns
    -------
    float
        Mean Squared Error.

    Examples
    --------
    >>> from mridc.collections.common.metrics.reconstruction_metrics import mse
    >>> import numpy as np
    >>> datax = np.random.rand(3, 100, 100)
    >>> datay = np.random.rand(3, 100, 100)
    >>> mse(datax, datay)
    0.17035991151556373
    """
    return np.mean((x - y) ** 2)  # type: ignore


def nmse(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Normalized Mean Squared Error (NMSE).

    Parameters
    ----------
    x : np.ndarray
        Target image. It must be a 3D array, where the first dimension is the number of slices. In case of 2D images,
        the first dimension should be 1.
    y : np.ndarray
        Predicted image. It must be a 3D array, where the first dimension is the number of slices. In case of 2D
        images, the first dimension should be 1.

    Returns
    -------
    float
        Normalized Mean Squared Error.

    Examples
    --------
    >>> from mridc.collections.common.metrics.reconstruction_metrics import nmse
    >>> import numpy as np
    >>> datax = np.random.rand(3, 100, 100)
    >>> datay = np.random.rand(3, 100, 100)
    >>> nmse(datax, datay)
    0.5001060028222054
    """
    return np.linalg.norm(x - y) ** 2 / np.linalg.norm(x) ** 2


def psnr(x: np.ndarray, y: np.ndarray, maxval: np.ndarray = None) -> float:
    """
    Compute Peak Signal to Noise Ratio (PSNR).

    Parameters
    ----------
    x : np.ndarray
        Target image. It must be a 3D array, where the first dimension is the number of slices. In case of 2D images,
        the first dimension should be 1.
    y : np.ndarray
        Predicted image. It must be a 3D array, where the first dimension is the number of slices. In case of 2D
        images, the first dimension should be 1.
    maxval : np.ndarray
        Maximum value of the images. If None, it is computed from the images. If the images are normalized, maxval
        should be 1.

    Returns
    -------
    float
        Peak Signal to Noise Ratio.

    Examples
    --------
    >>> from mridc.collections.reconstruction.metrics.reconstruction_metrics import psnr
    >>> import numpy as np
    >>> datax = np.random.rand(3, 100, 100)
    >>> datay = np.random.rand(3, 100, 100)
    >>> psnr(datax, datay)
    7.6700572264458

    .. note::
        x and y must be normalized to the same range, e.g. [0, 1].

        The PSNR is computed using the scikit-image implementation of the PSNR metric.
        Source: https://scikit-image.org/docs/dev/api/skimage.metrics.html#skimage.metrics.peak_signal_noise_ratio
    """
    if maxval is None:
        maxval = np.max(x)
    return peak_signal_noise_ratio(x, y, data_range=maxval)


def ssim(x: np.ndarray, y: np.ndarray, maxval: np.ndarray = None) -> float:
    """
    Compute Structural Similarity Index Measure (SSIM).

    Parameters
    ----------
    x : np.ndarray
        Target image. It must be a 3D array, where the first dimension is the number of slices. In case of 2D images,
        the first dimension should be 1.
    y : np.ndarray
        Predicted image. It must be a 3D array, where the first dimension is the number of slices. In case of 2D
        images, the first dimension should be 1.
    maxval : np.ndarray
        Maximum value of the images. If None, it is computed from the images. If the images are normalized, maxval
        should be 1.

    Returns
    -------
    float
        Structural Similarity Index Measure.

    Examples
    --------
    >>> from mridc.collections.common.metrics.reconstruction_metrics import ssim
    >>> import numpy as np
    >>> datax = np.random.rand(3, 100, 100)
    >>> datay = datax * 0.5
    >>> ssim(datax, datay)
    0.01833040155119426

    .. note::
        x and y must be normalized to the same range, e.g. [0, 1].

        The SSIM is computed using the scikit-image implementation of the SSIM metric.
        Source: https://scikit-image.org/docs/dev/api/skimage.metrics.html#skimage.metrics.structural_similarity
    """
    if x.ndim != 3:
        raise ValueError("Unexpected number of dimensions in ground truth.")
    if x.ndim != y.ndim:
        raise ValueError("Ground truth dimensions does not match prediction dimensions.")

    maxval = np.max(x) if maxval is None else maxval
    ssim_score = sum(
        structural_similarity(x[slice_num], y[slice_num], data_range=maxval) for slice_num in range(x.shape[0])
    )
    return ssim_score / x.shape[0]


METRIC_FUNCS = dict(MSE=mse, NMSE=nmse, PSNR=psnr, SSIM=ssim)


class ReconstructionMetrics:
    """
    Maintains running statistics for a given collection of reconstruction metrics.

    Examples
    --------
    >>> from mridc.collections.common.metrics.reconstruction_metrics import ReconstructionMetrics
    >>> import numpy as np
    >>> datax = np.random.rand(3, 100, 100)
    >>> datay = np.random.rand(3, 100, 100)
    >>> metrics = ReconstructionMetrics(METRIC_FUNCS, 'output', 'method')
    >>> metrics.push(datax, datay)
    >>> metrics.means()
    {'MSE': 0.17035991151556373, 'NMSE': 0.5001060028222054, 'PSNR': 7.6700572264458, 'SSIM': 0.01833040155119426}
    >>> metrics.__repr__()
    'MSE = 0.1704 +/- 0.01072 NMSE = 0.5001 +/- 0.01636 PSNR = 7.67 +/- 0.319 SSIM = 0.01833 +/- 0.03527\n'
    """

    def __init__(self, metric_funcs, output_path, method):
        """
        Parameters
        ----------
        metric_funcs : dict
            A dict where the keys are metric names and the values are Python functions for evaluating that metric.
        output_path : str
            Path to the output directory.
        method : str
            Reconstruction method.
        """
        self.metrics_scores = {metric: Statistics() for metric in metric_funcs}
        self.output_path = output_path
        self.method = method

    def push(self, x, y):
        """
        Pushes a new batch of metrics to the running statistics.

        Parameters
        ----------
        x : np.ndarray
            Target image. It must be a 3D array, where the first dimension is the number of slices. In case of 2D
            images, the first dimension should be 1.
        y : np.ndarray
            Predicted image. It must be a 3D array, where the first dimension is the number of slices. In case of 2D
            images, the first dimension should be 1.

        Returns
        -------
        dict
            A dict where the keys are metric names and the values are the computed metric scores.
        """
        for metric, func in METRIC_FUNCS.items():
            self.metrics_scores[metric].push(func(x, y))

    def means(self):
        """Mean of the means of each metric."""
        return {metric: stat.mean() for metric, stat in self.metrics_scores.items()}

    def stddevs(self):
        """Standard deviation of the means of each metric."""
        return {metric: stat.stddev() for metric, stat in self.metrics_scores.items()}

    def __repr__(self):
        """Representation of the metrics."""
        means = self.means()
        stddevs = self.stddevs()
        metric_names = sorted(list(means))

        res = " ".join(f"{name} = {means[name]:.4g} +/- {2 * stddevs[name]:.4g}" for name in metric_names) + "\n"
        with open(f"{self.output_path}metrics.txt", "a") as output:
            output.write(f"{self.method}: {res}")
        return res
