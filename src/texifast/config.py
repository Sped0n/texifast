from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

IMAGE_STD = [0.229, 0.224, 0.225]
IMAGE_MEAN = [0.485, 0.456, 0.406]


class TxfConfig:
    def __init__(
        self,
        size: tuple[int, int] = (420, 420),
        pad_fill_value: int = 255,
        rescale_factor: float = 0.00392156862745098,
        image_std: list[float] = IMAGE_STD,
        image_mean: list[float] = IMAGE_MEAN,
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        decoder_layers: int = 8,
    ) -> None:
        """
        Initialize the TxfConfig class.

        Parameters
        ----------
        size : tuple[int, int], optional
            Image size, defaults to (420, 420).
        pad_fill_value : float, optional
            Padding fill value, defaults to 255.0.
        rescale_factor : float, optional
            Rescale factor, defaults to 0.00392156862745098.
        image_std : list[float], optional
            Image standard deviation, defaults to [0.229, 0.224, 0.225].
        image_mean : list[float], optional
            Image mean, defaults to [0.485, 0.456, 0.406].
        bos_token : str, optional
            Beginning of sequence token, defaults to "<s>".
        eos_token : str, optional
            End of sequence token, defaults to "</s>".
        decoder_layers : int, optional
            Number of decoder layers, defaults to 8.
        """
        self.size: tuple[int, int] = size
        self.pad_fill_value: int = pad_fill_value
        self.rescale_factor: float = rescale_factor
        self.image_std: NDArray[np.float32] = np.array(image_std, dtype=np.float32)
        self.image_mean: NDArray[np.float32] = np.array(image_mean, dtype=np.float32)
        self.bos_token: str = bos_token
        self.eos_token: str = eos_token
        self.decoder_layers: int = decoder_layers
