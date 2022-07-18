# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

# Taken and adapted from https://github.com/bduffy0/motion-correction/blob/master/layer/motion_sim.py

import math
import random
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from mridc.collections.common.parts.utils import reshape_fortran


def get_center_rect(image: torch.tensor, center_percentage: float = 0.02, dim: int = 0) -> torch.tensor:
    """
    Get a center rectangle of a given dimension.

    Parameters
    ----------
    image : torch.tensor
        The image to get the center rectangle from.
    center_percentage : float
        The percentage of the image to take as the center rectangle.
    dim : int
        The dimension to take the center rectangle from.

    Returns
    -------
    torch.tensor
        The center rectangle.
    """
    shape = (image[0].item(), image[1].item())
    mask = torch.zeros(shape)
    half_pct = center_percentage / 2
    center = [int(x / 2) for x in shape]
    mask = torch.swapaxes(mask, 0, dim)
    mask[:, center[1] - math.ceil(shape[1] * half_pct) : math.ceil(center[1] + shape[1] * half_pct)] = 1
    mask = torch.swapaxes(mask, 0, dim)
    return mask


def segment_array_by_locs(shape: Sequence[int], locations: Sequence[int]) -> torch.tensor:
    """
    Generate a segmentation mask based on a list of locations.

    Parameters
    ----------
    shape : Sequence[int]
        The shape of the array to segment.
    locations : Sequence[int]
        The locations to segment the array into.

    Returns
    -------
    torch.tensor
        The segmentation mask.
    """
    mask_out = torch.zeros(torch.prod(shape), dtype=int)
    for i in range(len(locations) - 1):
        l = [locations[i], locations[i + 1]]
        mask_out[l[0] : l[1]] = i + 1
    return mask_out.reshape(shape)


def segments_to_random_indices(shape: Sequence[int], seg_lengths: Sequence[int]) -> torch.tensor:
    """
    Generate a segmentation mask based on a list of locations.

    Parameters
    ----------
    shape : Sequence[int]
        The shape of the array to segment.
    seg_lengths : Sequence[int]
        The lengths of the segments to generate.

    Returns
    -------
    torch.tensor
        The segmentation mask.
    """
    random_indices = torch.randint(low=0, high=shape, size=(sum(seg_lengths),)).sort()[0]
    seg_mask = torch.zeros(shape).type(torch.int)
    seg_new_indices = torch.cumsum(torch.tensor(seg_lengths), 0).tolist()
    seg_new_indices = [0] + seg_new_indices
    for i in range(len(seg_new_indices) - 1):
        seg_mask[random_indices[seg_new_indices[i] : seg_new_indices[i + 1]]] = i + 1
    return seg_mask


def segments_to_random_blocks(shape: Sequence[int], seg_lengths: Sequence[int]) -> torch.tensor:
    """
    Generate a segmentation mask based on a list of locations.

    Parameters
    ----------
    shape : Sequence[int]
        The shape of the array to segment.
    seg_lengths : Sequence[int]
        The lengths of the segments to generate.

    Returns
    -------
    torch.tensor
        The segmentation mask.
    """
    seg_mask = torch.zeros(shape).type(torch.int)
    seg_lengths_sorted = sorted(seg_lengths, reverse=True)
    for i, seg_len in enumerate(seg_lengths_sorted):
        loc = torch.randint(low=0, high=seg_mask.size()[0], size=(1,))
        while (sum(seg_mask[loc : loc + seg_len]) != 0) or (loc + seg_len > seg_mask.size()[0]):
            loc = torch.randint(low=0, high=seg_mask.size()[0], size=(1,))
        seg_mask[loc : loc + seg_len] = i + 1
    return seg_mask


def create_rand_partition(im_length: int, num_segments: int):
    """
    :param im_length: length of 1D array to partition
    :param num_segments: num segs to partition into
    :return: partition locations (list of indices)
    """
    # rand_segment_locs = sorted(np.random.randint(im_length, size=num_segments + 1).astype(list))
    rand_segment_locs = sorted(list(torch.randint(im_length, size=(num_segments + 1,))))
    rand_segment_locs[0] = 0
    rand_segment_locs[-1] = None
    return rand_segment_locs


def create_rotation_matrix_3d(angles: Sequence[float]) -> torch.tensor:
    """
    Create a 3D rotation matrix.

    Parameters
    ----------
    angles : Sequence[float]
        The angles to rotate the matrix by.

    Returns
    -------
    torch.tensor
        The rotation matrix.
    """
    mat1 = torch.FloatTensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(angles[0]), math.sin(angles[0])],
            [0.0, -math.sin(angles[0]), math.cos(angles[0])],
        ]
    )
    mat2 = torch.FloatTensor(
        [
            [math.cos(angles[1]), 0.0, -math.sin(angles[1])],
            [0.0, 1.0, 0.0],
            [math.sin(angles[1]), 0.0, math.cos(angles[1])],
        ]
    )
    mat3 = torch.FloatTensor(
        [
            [math.cos(angles[2]), math.sin(angles[2]), 0.0],
            [-math.sin(angles[2]), math.cos(angles[2]), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return (mat1 @ mat2) @ mat3


def translate_kspace(freq_domain: torch.tensor, translations: torch.tensor) -> torch.tensor:
    """
    Translate a k-space array.

    Parameters
    ----------
    freq_domain : torch.tensor
        The k-space array to translate.
    translations : torch.tensor
        The translations to apply to the k-space array.

    Returns
    -------
    torch.tensor
        The translated k-space array.
    """
    lin_spaces = [torch.linspace(-0.5, 0.5, x) for x in freq_domain.shape]
    meshgrids = torch.meshgrid(*lin_spaces, indexing="ij")
    grid_coords = torch.stack([mg.flatten() for mg in meshgrids], 0)
    phase_shift = torch.multiply(grid_coords, translations).sum(axis=0)  # phase shift is added
    exp_phase_shift = torch.exp(-2j * math.pi * phase_shift).to(freq_domain.device)

    # import matplotlib.pyplot as plt
    # import numpy as np
    #
    # _meshgrids = [torch.sum(x, 0).numpy() for x in meshgrids]
    # _translated_meshgrids = [x*t for x, t in zip(_meshgrids, torch.sum(translations, 0).cpu().numpy())]

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 2, 1, projection='3d')
    # ax.plot_surface(
    #     *_meshgrids,
    #     rstride=1,
    #     cstride=1,
    #     cmap='viridis',
    #     edgecolor='none',
    # )
    # # ax.set_yticks(y_tick * np.pi)
    # # ax.set_yticklabels(y_label, fontsize=8)
    #
    # ax.title.set_text('Original')
    #
    # # y_pi = _translated_meshgrids[1] / np.pi
    # # unit = 0.5
    # # y_tick = np.arange(-0.5, 0.5 + unit, unit)
    # # y_label = [r"$-\frac{\pi}{2}$", r"$0$", r"$+\frac{\pi}{2}$"]
    #
    # ax = fig.add_subplot(1, 2, 2, projection='3d')
    # ax.plot_surface(
    #     *_translated_meshgrids,
    #     rstride=1,
    #     cstride=1,
    #     cmap='viridis',
    #     edgecolor='none'
    # )
    # # ax.set_yticks(y_tick * np.pi)
    # # ax.set_yticklabels(y_label, fontsize=8)
    # ax.title.set_text('Translated')
    # plt.show()

    return exp_phase_shift


class MotionSimulation:
    """Simulates random translations and rotations in the frequency domain."""

    def __init__(
        self,
        type: str = "piecewise_transient",
        angle: float = 0,
        translation: float = 10,
        center_percentage: float = 0.02,
        motion_percentage: Sequence[float] = (15, 20),
        spatial_dims: Sequence[int] = (-2, -1),
        num_segments: int = 8,
        random_num_segments: bool = False,
        non_uniform: bool = False,
    ):
        """
        Initialize the motion simulation.

        Parameters
        ----------
        type : str
            The type of motion to simulate.
        angle : float
            The angle to rotate the k-space array by.
        translation : float
            The translation to apply to the k-space array.
        center_percentage : float
            The percentage of the k-space array to center the motion.
        motion_percentage : Sequence[float]
            The percentage of the k-space array to apply the motion.
        spatial_dims : Sequence[int]
            The spatial dimensions to apply the motion to.
        num_segments : int
            The number of segments to partition the k-space array into.
        random_num_segments : bool
            Whether to randomly generate the number of segments.
        non_uniform : bool
            Whether to use non-uniform sampling.
        """
        self.type = type
        self.angle, self.translation = angle, translation
        self.center_percentage = center_percentage

        if motion_percentage[1] == motion_percentage[0]:
            motion_percentage[1] += 1  # type: ignore
        elif motion_percentage[1] < motion_percentage[0]:
            raise ValueError("Uniform is not defined when low>= high.")

        self.motion_percentage = motion_percentage

        self.spatial_dims = spatial_dims
        self._spatial_dims = random.choice(spatial_dims)

        self.num_segments = num_segments
        self.random_num_segments = random_num_segments

        if non_uniform:
            raise NotImplementedError("NUFFT is not implemented. This is a feature to be added in the future.")

        self.trajectory = None
        self.params: Dict[Any, Any] = {}

    def _calc_dimensions(self, shape):
        """
        Calculate the dimensions to apply the motion to.

        Parameters
        ----------
        shape : Sequence[int]
            The shape of the image.

        Returns
        -------
        Sequence[int]
            The dimensions to apply the motion to.
        """
        pe_dims = [0, 1, 2]
        pe_dims.pop(self._spatial_dims)
        self.phase_encoding_dims = pe_dims
        shape = list(shape)
        self.shape = shape.copy()
        shape.pop(self._spatial_dims)
        self.phase_encoding_shape = torch.tensor(shape)
        self.num_phase_encoding_steps = self.phase_encoding_shape[0] * self.phase_encoding_shape[1]
        self._spatial_dims = len(self.shape) - 1 if self._spatial_dims == -1 else self._spatial_dims

    def _simulate_random_trajectory(self):
        """
        Simulate a random trajectory.

        Returns
        -------
        torch.tensor
            The random trajectory.
        """
        pct_corrupt = torch.distributions.Uniform(*[x / 100 for x in self.motion_percentage]).sample((1, 1))

        corrupt_matrix_shape = torch.tensor([int(x * math.sqrt(pct_corrupt)) for x in self.phase_encoding_shape])

        if torch.prod(corrupt_matrix_shape) == 0:
            corrupt_matrix_shape = [1, 1]

        if self.type in {"gaussian"}:
            num_segments = torch.prod(corrupt_matrix_shape)
        else:
            if not self.random_num_segments:
                num_segments = self.num_segments
            else:
                num_segments = random.randint(1, self.num_segments)

        # segment a smaller vector occupying pct_corrupt percent of the space
        if self.type in {"piecewise_transient", "piecewise_constant"}:
            seg_locs = create_rand_partition(torch.prod(corrupt_matrix_shape), num_segments=num_segments)
        else:
            seg_locs = list(range(num_segments))

        rand_segmentation = segment_array_by_locs(shape=torch.prod(corrupt_matrix_shape), locations=seg_locs)

        seg_lengths = [(rand_segmentation == seg_num).sum() for seg_num in torch.unique(rand_segmentation)]

        # assign segments to a vector with same number of elements as pe-steps
        if self.type in {"piecewise_transient", "gaussian"}:
            seg_vector = segments_to_random_indices(torch.prod(self.phase_encoding_shape), seg_lengths)
        else:
            seg_vector = segments_to_random_blocks(torch.prod(self.phase_encoding_shape), seg_lengths)

        # reshape to phase encoding shape with a random order
        reshape_order = random.choice(["F", "C"])

        if reshape_order == "F":
            seg_array = reshape_fortran(
                seg_vector, (self.phase_encoding_shape[0].item(), self.phase_encoding_shape[1].item())
            )
        else:
            seg_array = seg_vector.reshape((self.phase_encoding_shape[0].item(), self.phase_encoding_shape[1].item()))

        self.order = reshape_order

        # mask center k-space
        mask_not_including_center = (
            get_center_rect(
                self.phase_encoding_shape,
                center_percentage=self.center_percentage,
                dim=1 if reshape_order == "C" else 0,
            )
            == 0
        )

        seg_array = seg_array * mask_not_including_center

        # generate random translations and rotations
        rand_translations = torch.distributions.normal.Normal(loc=0, scale=self.translation).sample(
            (num_segments + 1, 3)
        )
        rand_rotations = torch.distributions.normal.Normal(loc=0, scale=self.angle).sample((num_segments + 1, 3))

        # if segment==0, then no motion
        rand_translations[0, :] = 0
        rand_rotations[0, :] = 0

        # lookup values for each segment
        translations_pe = [rand_translations[:, i][seg_array.long()] for i in range(3)]
        rotations_pe = [rand_rotations[:, i][seg_array.long()] for i in range(3)]

        # reshape and convert to radians
        translations = torch.stack(
            [torch.broadcast_to(x.unsqueeze(self._spatial_dims), self.shape) for x in translations_pe], 0
        )
        rotations = torch.stack(
            [torch.broadcast_to(x.unsqueeze(self._spatial_dims), self.shape) for x in rotations_pe], 0
        )

        rotations = rotations * (math.pi / 180.0)  # convert to radians

        translations = translations.reshape(3, -1)
        rotations = rotations.reshape(3, -1).reshape(3, -1)

        return translations, rotations

    def forward(self, kspace, translations_rotations=None) -> torch.Tensor:
        """
        Apply the motion to the kspace.

        Parameters
        ----------
        kspace : torch.Tensor
            The kspace to apply the motion to.
        translations_rotations : Optional[Tuple[torch.Tensor, torch.Tensor]]
            The translations and rotations to apply. If None, then the motion is simulated.

        Returns
        -------
        torch.Tensor
            The kspace with the motion applied.
        """
        if kspace.shape[-1] == 2:
            kspace = torch.view_as_complex(kspace)

        self._calc_dimensions(kspace.shape)
        translations, rotations = (
            self._simulate_random_trajectory() if translations_rotations is None else translations_rotations
        )

        self.params["translations"] = translations
        self.params["rotations"] = rotations

        exp_phase_shift = translate_kspace(freq_domain=kspace, translations=rotations)

        return exp_phase_shift

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
