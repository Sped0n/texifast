from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

IMAGE_STD: list[float] = [0.229, 0.224, 0.225]
IMAGE_MEAN: list[float] = [0.485, 0.456, 0.406]


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
        """Initialize the TxfConfig class.

        Note:
            The default values are set based on the `preprocess_config.json` and `config.json` in
            huggingface `Spedon/texify-quantized-onnx` repository.

        Args:
            size (tuple[int, int], optional): Size of the input image. Defaults to (420, 420).
            pad_fill_value (int, optional): Fill value for padding. Defaults to 255.
            rescale_factor (float, optional): Rescale factor for the input image. Defaults to 0.00392156862745098.
            image_std (list[float], optional): Standard deviation of the input image. Defaults to IMAGE_STD.
            image_mean (list[float], optional): Mean of the input image. Defaults to IMAGE_MEAN.
            bos_token (str, optional): Beginning of sentence token. Defaults to "<s>".
            eos_token (str, optional): End of sentence token. Defaults to "</s>".
            decoder_layers (int, optional): Number of decoder layers. Defaults to 8.
        """
        self.size: tuple[int, int] = size
        self.pad_fill_value: int = pad_fill_value
        self.rescale_factor: float = rescale_factor
        self.image_std: NDArray[np.float32] = np.array(image_std, dtype=np.float32)
        self.image_mean: NDArray[np.float32] = np.array(image_mean, dtype=np.float32)
        self.bos_token: str = bos_token
        self.eos_token: str = eos_token
        self.decoder_layers: int = decoder_layers
