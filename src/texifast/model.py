from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import onnxruntime as ort

from .helpers import logger

if TYPE_CHECKING:
    from collections.abc import Sequence
    from os import PathLike
    from typing import Any, TypeAlias

    from numpy.typing import NDArray

    TxfArray: TypeAlias = NDArray[np.float32 | np.float16]


class TxfModel:
    def __init__(
        self,
        encoder_model_path: str | bytes | PathLike[Any],
        decoder_model_path: str | bytes | PathLike[Any],
        provider: Sequence[str | tuple[str, dict[Any, Any]]] | None = None,
        session_options: ort.SessionOptions  # pyright: ignore[reportUnknownParameterType]
        | None = None,
        provider_options: Sequence[dict[Any, Any]] | None = None,
        use_io_binding: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the TxfModel class.

        Parameters
        ----------
        encoder_model_path : str | bytes | PathLike
            Path to the encoder model.
        decoder_model_path : str | bytes | PathLike
            Path to the decoder model.
        provider : Sequence[str | tuple[str, dict[Any, Any]]], optional
            Execution provider to use. Defaults to None.
        session_options : ort.SessionOptions, optional
            ONNXRuntime session options. Defaults to None.
        provider_options : Sequence[dict[Any, Any]], optional
            Execution provider options. Defaults to None.
        use_io_binding : bool, optional
            Enable I/O binding. Defaults to None.

        Raises
        ------
        ONNXRuntimeError
            ONNXRuntime exception.
        ValueError
            Encoder and decoder models type mismatch.
        NotImplementedError
            Unsupported dtype.
        """
        # inference sessions
        self.encoder_session = ort.InferenceSession(
            encoder_model_path,
            sess_options=session_options,
            providers=provider,
            provider_options=provider_options,
            **kwargs,
        )
        self.decoder_session = ort.InferenceSession(
            decoder_model_path,
            sess_options=session_options,
            providers=provider,
            provider_options=provider_options,
            **kwargs,
        )
        # device type
        providers: list[str] = self.encoder_session.get_providers()
        self.device_type = "cuda" if "CUDAExecutionProvider" in providers else "cpu"
        logger.info(f"Device type: {self.device_type}")
        # warn user if running a quantized model on CUDAExecutionProvider
        model_meta: ort.ModelMetadata = self.encoder_session.get_modelmeta()
        if "quant" in model_meta.producer_name and "CUDAExecutionProvider" in providers:
            logger.warning(
                """
                If you are running a quantized model on CUDAExecutionProvider, you may experience unexpected results or extreme bad performance.
                It is recommended to use float32 or float16 models with CUDAExecutionProvider.
                """
            )
        # dtype
        encoder_dtype: str = self.encoder_session.get_outputs()[0].type
        decoder_dtype: str = self.decoder_session.get_outputs()[0].type
        if encoder_dtype != decoder_dtype:
            err_msg: str = f"Encoder and decoder models type mismatch, encoder: {encoder_dtype}, decoder: {decoder_dtype}"
            logger.error(err_msg)
            raise ValueError(err_msg)
        if self.encoder_session.get_inputs()[0].type == "tensor(float16)":
            self.dtype = np.float16
            logger.info("Using mixed precision model")
        elif self.encoder_session.get_inputs()[0].type == "tensor(float)":
            self.dtype = np.float32
        else:
            err_msg = f"Unsupported dtype: {self.encoder_session.get_inputs()[0].type}"
            logger.error(err_msg)
            raise NotImplementedError(err_msg)
        # io binding
        if use_io_binding is not None:
            self.use_io_binding: bool = use_io_binding
            logger.info(
                f"I/O binding {'enabled' if self.use_io_binding else 'disabled'}"
            )
        else:
            self.use_io_binding = True if self.device_type == "cuda" else False
            logger.info(
                f"I/O binding not specified, using default `{self.use_io_binding}` for `{self.device_type}` device type"
            )
