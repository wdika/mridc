# encoding: utf-8
# Parts of the code have been taken from https://github.com/facebookresearch/fastMRI
__author__ = "Dimitrios Karkalousos"

import contextlib
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch


@contextlib.contextmanager
def temp_seed(rng: np.random, seed: Optional[Union[int, Tuple[int, ...]]]):
    """
    Temporarily sets the seed of the given random number generator.

    Parameters
    ----------
    rng: The random number generator.
    seed: The seed to set.

    Returns
    -------
    A context manager.
    """
    if seed is None:
        try:
            yield
        finally:
            pass
    else:
        state = rng.get_state()
        rng.seed(seed)
        try:
            yield
        finally:
            rng.set_state(state)


class MaskFunc:
    """A class that defines a mask function."""

    def __init__(self, center_fractions: Sequence[float], accelerations: Sequence[int]):
        """
        Initialize the mask function.

        Parameters
        ----------
        center_fractions: Fraction of low-frequency columns to be retained. If multiple values are provided, then \
        one of these numbers is chosen uniformly each time. For 2D setting this value corresponds to setting the \
        Full-Width-Half-Maximum.
        accelerations: Amount of under-sampling. This should have the same length as center_fractions. If multiple \
        values are provided, then one of these is chosen uniformly each time.
        """
        if len(center_fractions) != len(accelerations):
            raise ValueError("Number of center fractions should match number of accelerations")

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.rng = np.random.RandomState()  # pylint: disable=no-member

    def __call__(
        self,
        shape: Sequence[int],
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
        half_scan_percentage: Optional[float] = 0.0,
        scale: Optional[float] = 0.02,
    ) -> Tuple[torch.Tensor, int]:
        """

        Parameters
        ----------
        shape: Shape of the input tensor.
        seed: Seed for the random number generator.
        half_scan_percentage: Percentage of the low-frequency columns to be retained.
        scale: Scale of the mask.

        Returns
        -------
        A tuple of the mask and the number of low-frequency columns retained.
        """
        raise NotImplementedError

    def choose_acceleration(self):
        """Choose acceleration."""
        choice = self.rng.randint(0, len(self.accelerations))
        center_fraction = self.center_fractions[choice]
        acceleration = self.accelerations[choice]

        return center_fraction, acceleration


class RandomMaskFunc(MaskFunc):
    """
    RandomMaskFunc creates a sub-sampling mask of a given shape.

    The mask selects a subset of columns from the input k-space data. If the k-space data has N columns, the mask \
    picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center corresponding to low-frequencies.
        2. The other columns are selected uniformly at random with a probability equal to: \
        prob = (N / acceleration - N_low_freqs) /  (N - N_low_freqs). This ensures that the expected number of \
        columns selected is equal to (N / acceleration).

    It is possible to use multiple center_fractions and accelerations, in which case one possible (center_fraction, \
    acceleration) is chosen uniformly at random each time the RandomMaskFunc object is called.

    For example, if accelerations = [4, 8] and center_fractions = [0.08, 0.04], then there is a 50% probability that \
    4-fold acceleration with 8% center  fraction is selected and a 50% probability that 8-fold acceleration with 4% \
    center fraction is selected.
    """

    def __call__(
        self,
        shape: Sequence[int],
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
        half_scan_percentage: Optional[float] = 0.0,
        scale: Optional[float] = 0.02,
    ) -> Tuple[torch.Tensor, int]:
        """
        Parameters
        ----------
        shape: The shape of the mask to be created. The shape should have at least 3 dimensions. Samples are drawn \
        along the second last dimension.
        seed: Seed for the random number generator. Setting the seed ensures the same mask is generated each time \
        for the same shape. The random state is reset afterwards.
        half_scan_percentage: Optional; Defines a fraction of the k-space data that is not sampled.
        scale: Optional; Defines the scale of the center of the mask.

        Returns
        -------
        A tuple of the mask and the number of columns selected.
        """
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        with temp_seed(self.rng, seed):
            num_cols = shape[-2]
            center_fraction, acceleration = self.choose_acceleration()

            # create the mask
            num_low_freqs = int(round(num_cols * center_fraction))
            prob = (num_cols / acceleration - num_low_freqs) / (num_cols - num_low_freqs)
            mask = self.rng.uniform(size=num_cols) < prob  # type: ignore
            pad = torch.div((num_cols - num_low_freqs + 1), 2, rounding_mode="trunc").item()
            mask[pad : pad + num_low_freqs] = True

            # reshape the mask
            mask_shape = [1 for _ in shape]
            mask_shape[-2] = num_cols
            mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

        return mask, acceleration


class EquispacedMaskFunc(MaskFunc):
    """
    EquispacedMaskFunc creates a sub-sampling mask of a given shape.

    The mask selects a subset of columns from the input k-space data. If the k-space data has N columns, the mask \
    picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center corresponding to low-frequencies.
        2. The other columns are selected with equal spacing at a proportion that reaches the desired acceleration \
        rate taking into consideration the number of low frequencies. This ensures that the expected number of \
        columns selected is equal to (N / acceleration)

    It is possible to use multiple center_fractions and accelerations, in which case one possible (center_fraction, \
    acceleration) is chosen uniformly at random each time the EquispacedMaskFunc object is called.

    Note that this function may not give equispaced samples (documented in \
    https://github.com/facebookresearch/fastMRI/issues/54), which will require modifications to standard GRAPPA \
    approaches. Nonetheless, this aspect of the function has been preserved to match the public multicoil data.
    """

    def __call__(
        self,
        shape: Sequence[int],
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
        half_scan_percentage: Optional[float] = 0.0,
        scale: Optional[float] = 0.02,
    ) -> Tuple[torch.Tensor, int]:
        """
        Parameters
        ----------
        shape: The shape of the mask to be created. The shape should have at least 3 dimensions. Samples are drawn \
        along the second last dimension.
        seed: Seed for the random number generator. Setting the seed ensures the same mask is generated each time for \
        the same shape. The random state is reset afterwards.
        half_scan_percentage: Optional; Defines a fraction of the k-space data that is not sampled.
        scale: Optional; Defines the scale of the center of the mask.

        Returns
        -------
        A tuple of the mask and the number of columns selected.
        """
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        with temp_seed(self.rng, seed):
            center_fraction, acceleration = self.choose_acceleration()
            num_cols = shape[-2]
            num_low_freqs = int(round(num_cols * center_fraction))

            # create the mask
            mask = np.zeros(num_cols, dtype=np.float32)
            pad = torch.div((num_cols - num_low_freqs + 1), 2, rounding_mode="trunc").item()
            mask[pad : pad + num_low_freqs] = True  # type: ignore

            # determine acceleration rate by adjusting for the number of low frequencies
            adjusted_accel = (acceleration * (num_low_freqs - num_cols)) / (num_low_freqs * acceleration - num_cols)
            offset = self.rng.randint(0, round(adjusted_accel))

            accel_samples = np.arange(offset, num_cols - 1, adjusted_accel)
            accel_samples = np.around(accel_samples).astype(np.uint)
            mask[accel_samples] = True

            # reshape the mask
            mask_shape = [1 for _ in shape]
            mask_shape[-2] = num_cols
            mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

        return mask, acceleration


class Gaussian1DMaskFunc(MaskFunc):
    """
    Creates a 1D sub-sampling mask of a given shape.

    For autocalibration purposes, data points near the k-space center will be fully sampled within an ellipse of \
    which the half-axes will set to the set scale % of the fully sampled region. The remaining points will be sampled \
    according to a Gaussian distribution.

    The center fractions here act as Full-Width at Half-Maximum (FWHM) values.
    """

    def __call__(
        self,
        shape: Union[Sequence[int], np.ndarray],
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
        half_scan_percentage: Optional[float] = 0.0,
        scale: Optional[float] = 0.02,
    ) -> Tuple[torch.Tensor, int]:
        """
        Parameters
        ----------
        shape: The shape of the mask to be created. The shape should have at least 3 dimensions. Samples are drawn \
        along the second last dimension.
        seed: Seed for the random number generator. Setting the seed ensures the same mask is generated each time \
        for the same shape. The random state is reset afterwards.
        half_scan_percentage: Optional; Defines a fraction of the k-space data that is not sampled.
        scale: For autocalibration purposes, data points near the k-space center will be fully sampled within an \
        ellipse of which the half-axes will set to the set scale % of the fully sampled region

        Returns
        -------
        A tuple of the mask and the number of columns selected.
        """
        dims = [1 for _ in shape]
        self.shape = tuple(shape[-3:-1])
        dims[-2] = self.shape[-1]

        full_width_half_maximum, acceleration = self.choose_acceleration()
        if not isinstance(full_width_half_maximum, list):
            full_width_half_maximum = [full_width_half_maximum] * 2
        self.full_width_half_maximum = full_width_half_maximum
        self.acceleration = acceleration

        self.scale = scale

        mask = self.gaussian_kspace()
        mask[tuple(self.gaussian_coordinates())] = 1.0

        mask = np.fft.ifftshift(np.fft.ifftshift(np.fft.ifftshift(mask, axes=0), axes=0), axes=(0, 1))

        if half_scan_percentage != 0:
            mask[: int(np.round(mask.shape[0] * half_scan_percentage)), :] = 0.0

        return torch.from_numpy(mask[0].reshape(dims).astype(np.float32)), acceleration

    def gaussian_kspace(self):
        """Creates a Gaussian sampled k-space center."""
        scaled = int(self.shape[0] * self.scale)
        center = np.ones((scaled, self.shape[1]))
        top_scaled = torch.div((self.shape[0] - scaled), 2, rounding_mode="trunc").item()
        bottom_scaled = self.shape[0] - scaled - top_scaled
        top = np.zeros((top_scaled, self.shape[1]))
        btm = np.zeros((bottom_scaled, self.shape[1]))
        return np.concatenate((top, center, btm))

    def gaussian_coordinates(self):
        """Creates a Gaussian sampled k-space coordinates."""
        n_sample = int(self.shape[0] / self.acceleration)
        kernel = self.gaussian_kernel()
        idxs = np.random.choice(range(self.shape[0]), size=n_sample, replace=False, p=kernel)
        xsamples = np.concatenate([np.tile(i, self.shape[1]) for i in idxs])
        ysamples = np.concatenate([range(self.shape[1]) for _ in idxs])
        return xsamples, ysamples

    def gaussian_kernel(self):
        """Creates a Gaussian sampled k-space kernel."""
        kernel = 1
        for fwhm, kern_len in zip(self.full_width_half_maximum, self.shape):
            sigma = fwhm / np.sqrt(8 * np.log(2))
            x = np.linspace(-1.0, 1.0, kern_len)
            g = np.exp(-(x**2 / (2 * sigma**2)))
            kernel = g
            break
        kernel = kernel / kernel.sum()
        return kernel


class Gaussian2DMaskFunc(MaskFunc):
    """
    Creates a 2D sub-sampling mask of a given shape.

    For autocalibration purposes, data points near the k-space center will be fully sampled within an ellipse of \
    which the half-axes will set to the set scale % of the fully sampled region. The remaining points will be sampled \
    according to a Gaussian distribution.

    The center fractions here act as Full-Width at Half-Maximum (FWHM) values.
    """

    def __call__(
        self,
        shape: Union[Sequence[int], np.ndarray],
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
        half_scan_percentage: Optional[float] = 0.0,
        scale: Optional[float] = 0.02,
    ) -> Tuple[torch.Tensor, int]:
        """
        Parameters
        ----------
        shape: The shape of the mask to be created. The shape should have at least 3 dimensions. Samples are drawn \
        along the second last dimension.
        seed: Seed for the random number generator. Setting the seed ensures the same mask is generated each time for \
         the same shape. The random state is reset afterwards.
        half_scan_percentage: Optional; Defines a fraction of the k-space data that is not sampled.
        scale: For autocalibration purposes, data points near the k-space center will be fully sampled within an \
        ellipse of which the half-axes will set to the set scale % of the fully sampled region

        Returns
        -------
        A tuple of the mask and the number of columns selected.
        """
        dims = [1 for _ in shape]
        self.shape = tuple(shape[-3:-1])
        dims[-3:-1] = self.shape

        full_width_half_maximum, acceleration = self.choose_acceleration()

        if not isinstance(full_width_half_maximum, list):
            full_width_half_maximum = [full_width_half_maximum] * 2
        self.full_width_half_maximum = full_width_half_maximum

        self.acceleration = acceleration
        self.scale = scale

        mask = self.gaussian_kspace()
        mask[tuple(self.gaussian_coordinates())] = 1.0

        if half_scan_percentage != 0:
            mask[: int(np.round(mask.shape[0] * half_scan_percentage)), :] = 0.0

        return torch.from_numpy(mask.reshape(dims).astype(np.float32)), acceleration

    def gaussian_kspace(self):
        """Creates a Gaussian sampled k-space center."""
        a, b = self.scale * self.shape[0], self.scale * self.shape[1]
        afocal, bfocal = self.shape[0] / 2, self.shape[1] / 2
        xx, yy = np.mgrid[: self.shape[0], : self.shape[1]]
        ellipse = np.power((xx - afocal) / a, 2) + np.power((yy - bfocal) / b, 2)
        return (ellipse < 1).astype(float)

    def gaussian_coordinates(self):
        """Creates a Gaussian sampled k-space coordinates."""
        n_sample = int(self.shape[0] * self.shape[1] / self.acceleration)
        cartesian_prod = list(np.ndindex(self.shape))  # type: ignore
        kernel = self.gaussian_kernel()
        idxs = np.random.choice(range(len(cartesian_prod)), size=n_sample, replace=False, p=kernel.flatten())
        return list(zip(*list(map(cartesian_prod.__getitem__, idxs))))

    def gaussian_kernel(self):
        """Creates a Gaussian kernel."""
        kernels = []
        for fwhm, kern_len in zip(self.full_width_half_maximum, self.shape):
            sigma = fwhm / np.sqrt(8 * np.log(2))
            x = np.linspace(-1.0, 1.0, kern_len)
            g = np.exp(-(x**2 / (2 * sigma**2)))
            kernels.append(g)
        kernel = np.sqrt(np.outer(kernels[0], kernels[1]))
        kernel = kernel / kernel.sum()
        return kernel


class Poisson2DMaskFunc(MaskFunc):
    """
    Creates a 2D sub-sampling mask of a given shape.

    For autocalibration purposes, data points near the k-space center will be fully sampled within an ellipse of \
    which the half-axes will set to the set scale % of the fully sampled region. The remaining points will be sampled \
     according to a (variable density) Poisson distribution.

    For a given acceleration factor to be accurate, the scale for the fully sampled center should remain at the \
    default 0.02. A predefined list is used to convert the acceleration factor to the appropriate r parameter needed \
    for the variable density calculation. This list has been made to accommodate acceleration factors of 4 up to 21, \
    rounding off to the nearest one available. As such, acceleration factors outside this range cannot be used.
    """

    def __call__(
        self,
        shape: Union[Sequence[int], np.ndarray],
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
        half_scan_percentage: Optional[float] = 0.0,
        scale: Optional[float] = 0.02,
    ) -> Tuple[torch.Tensor, int]:
        """
        Parameters
        ----------
        shape: The shape of the mask to be created. The shape should have at least 3 dimensions. Samples are drawn \
        along the second last dimension.
        seed: Seed for the random number generator. Setting the seed ensures the same mask is generated each time \
        for the same shape. The random state is reset afterwards.
        half_scan_percentage: Optional; Defines a fraction of the k-space data that is not sampled.
        scale: For autocalibration purposes, data points near the k-space center will be fully sampled within an \
        ellipse of which the half-axes will set to the set scale % of the fully sampled region

        Returns
        -------
        A tuple of the mask and the number of columns selected.
        """
        dims = [1 for _ in shape]
        self.shape = tuple(shape[-3:-1])
        dims[-3:-1] = self.shape

        _, acceleration = self.choose_acceleration()
        if acceleration > 21.5 or acceleration < 3.5:
            raise ValueError(f"Acceleration {acceleration} is not supported for Poisson 2D masking.")

        self.acceleration = acceleration
        self.scale = scale

        # TODO: consider moving this to a yaml file
        rfactor = [
            21.22,
            20.32,
            19.06,
            18.22,
            17.41,
            16.56,
            15.86,
            15.12,
            14.42,
            13.88,
            13.17,
            12.76,
            12.21,
            11.72,
            11.09,
            10.68,
            10.35,
            10.02,
            9.61,
            9.22,
            9.03,
            8.66,
            8.28,
            8.1,
            7.74,
            7.62,
            7.32,
            7.04,
            6.94,
            6.61,
            6.5,
            6.27,
            6.15,
            5.96,
            5.83,
            5.59,
            5.46,
            5.38,
            5.15,
            5.05,
            4.9,
            4.86,
            4.67,
            4.56,
            4.52,
            4.41,
            4.31,
            4.21,
            4.11,
            3.99,
        ]
        self.r = min(range(len(rfactor)), key=lambda i: abs(rfactor[i] - self.acceleration)) + 40

        pattern1 = self.poisson_disc2d()
        pattern2 = self.centered_circle()
        mask = np.logical_or(pattern1, pattern2)

        if half_scan_percentage != 0:
            mask[: int(np.round(mask.shape[0] * half_scan_percentage)), :] = 0.0

        return (torch.from_numpy(mask.reshape(dims).astype(np.float32)), acceleration)

    def poisson_disc2d(self):
        """Creates a 2D Poisson disc pattern."""
        # Amount of tries before discarding a reference point for new samples
        k = 10

        # Amount of samples to be drawn
        pattern_shape = (self.shape[0] - 1, self.shape[1] - 1)

        # Initialize the pattern
        center = np.array([1.0 * pattern_shape[0] / 2, 1.0 * pattern_shape[1] / 2])
        width, height = pattern_shape

        # Cell side length (equal to r_min)
        a = 1

        # Number of cells in the x- and y-directions of the grid
        nx, ny = int(width / a), int(height / a)

        # A list of coordinates in the grid of cells
        coords_list = [(ix, iy) for ix in range(nx + 1) for iy in range(ny + 1)]

        # Initialize the dictionary of cells: each key is a cell's coordinates, the corresponding value is the index
        # of that cell's point's that might cause conflict when adding a new point.
        cells = {coords: [] for coords in coords_list}
        centernorm = np.linalg.norm(center)

        def calc_r(coords):
            """Calculate r for the given coordinates."""
            return ((np.linalg.norm(np.asarray(coords) - center) / centernorm) * 240 + 50) / self.r

        def get_cell_coords(pt):
            """Get the coordinates of the cell that pt = (x,y) falls in."""
            return int(np.floor_divide(pt[0], a)), int(np.floor_divide(pt[1], a))

        def mark_neighbours(idx):
            """Add sample index to the cells within r(point) range of the point."""
            coords = samples[idx]
            if idx in cells[get_cell_coords(coords)]:
                # This point is already marked on the grid, so we can skip
                return

            # Mark the point on the grid
            rx = calc_r(coords)
            xvals = np.arange(coords[0] - rx, coords[0] + rx)
            yvals = np.arange(coords[1] - rx, coords[1] + rx)

            # Get the coordinates of the cells that the point falls in
            xvals = xvals[(xvals >= 0) & (xvals <= width)]
            yvals = yvals[(yvals >= 0) & (yvals <= height)]

            def dist(x, y):
                """Calculate the distance between the point and the cell."""
                return np.sqrt((coords[0] - x) ** 2 + (coords[1] - y) ** 2) < rx

            xx, yy = np.meshgrid(xvals, yvals, sparse=False)

            # Mark the points in the grid
            pts = np.vstack((xx.ravel(), yy.ravel())).T
            pts = pts[dist(pts[:, 0], pts[:, 1])]

            return [cells[get_cell_coords(pt)].append(idx) for pt in pts]

        def point_valid(pt):
            """Check if the point is valid."""
            rx = calc_r(pt)
            if rx < 1:
                if np.linalg.norm(pt - center) < self.scale * width:
                    return False
                rx = 1

            # Get the coordinates of the cells that the point falls in
            neighbour_idxs = cells[get_cell_coords(pt)]
            for n in neighbour_idxs:
                n_coords = samples[n]

                # Squared distance between or candidate point, pt, and this nearby_pt.
                distance = np.sqrt((n_coords[0] - pt[0]) ** 2 + (n_coords[1] - pt[1]) ** 2)
                if distance < rx:
                    # The points are too close, so pt is not a candidate.
                    return False

            # All points tested: if we're here, pt is
            return True

        def get_point(k, refpt):
            """
            Try to find a candidate point relative to refpt to emit in the sample. We draw up to k points from the
            annulus of inner radius r, outer radius 2r around the reference point, refpt. If none of them are suitable
            return False. Otherwise, return the pt.
            """
            i = 0
            rx = calc_r(refpt)
            while i < k:
                rho, theta = np.random.uniform(rx, 2 * rx), np.random.uniform(0, 2 * np.pi)
                pt = refpt[0] + rho * np.cos(theta), refpt[1] + rho * np.sin(theta)
                if not (0 < pt[0] < width and 0 < pt[1] < height):
                    # Off the grid, try again.
                    continue
                if point_valid(pt):
                    return pt
                i += 1

            # We failed to find a suitable point in the vicinity of refpt.
            return False

        # Pick a random point to start with.
        pt = (np.random.uniform(0, width), np.random.uniform(0, height))
        samples = [pt]
        cursample = 0
        mark_neighbours(0)

        # Set active, in the sense that we're going to look for more points in its neighbourhood.
        active = [0]

        # As long as there are points in the active list, keep trying to find samples.
        while active:
            # choose a random "reference" point from the active list.
            idx = np.random.choice(active)
            refpt = samples[idx]

            # Try to pick a new point relative to the reference point.
            pt = get_point(k, refpt)
            if pt:
                # Point pt is valid: add it to the samples list and mark it as active
                samples.append(pt)
                cursample += 1
                active.append(cursample)
                mark_neighbours(cursample)
            else:
                # We had to give up looking for valid points near refpt, so remove it from the list of "active" points.
                active.remove(idx)

        samples = np.rint(np.array(samples)).astype(int)
        samples = np.unique(samples[:, 0] + 1j * samples[:, 1])
        samples = np.column_stack((samples.real, samples.imag)).astype(int)

        poisson_pattern = np.zeros((pattern_shape[0] + 1, pattern_shape[1] + 1), dtype=bool)
        poisson_pattern[samples[:, 0], samples[:, 1]] = True

        return poisson_pattern

    def centered_circle(self):
        """Creates a boolean centered circle image using the scale as a radius."""
        center_x = int((self.shape[0] - 1) / 2)
        center_y = int((self.shape[1] - 1) / 2)

        X, Y = np.indices(self.shape)
        radius = int(self.shape[0] * self.scale)
        return ((X - center_x) ** 2 + (Y - center_y) ** 2) < radius**2


def create_mask_for_mask_type(
    mask_type_str: str, center_fractions: Sequence[float], accelerations: Sequence[int]
) -> MaskFunc:
    """
    Creates a MaskFunc object for the given mask type.

    Parameters
    ----------
    mask_type_str: The string representation of the mask type.
    center_fractions: The center fractions for the mask.
    accelerations: The accelerations for the mask.

    Returns
    -------
    A MaskFunc object.
    """
    if mask_type_str == "random1d":
        return RandomMaskFunc(center_fractions, accelerations)
    if mask_type_str == "equispaced1d":
        return EquispacedMaskFunc(center_fractions, accelerations)
    if mask_type_str == "gaussian1d":
        return Gaussian1DMaskFunc(center_fractions, accelerations)
    if mask_type_str == "gaussian2d":
        return Gaussian2DMaskFunc(center_fractions, accelerations)
    if mask_type_str == "poisson2d":
        return Poisson2DMaskFunc(center_fractions, accelerations)
    raise NotImplementedError(f"{mask_type_str} not supported")
