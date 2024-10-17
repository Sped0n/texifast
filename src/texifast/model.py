from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import onnxruntime as ort

from .helpers import logger

if TYPE_CHECKING:
    from collections.abc import Sequence
    from os import PathLike
    from typing import Any


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
        """Initialize the TxfModel class.

        Note:
            You should pass in same type(quantized/fp16) of encoder and decoder models, do not mix them. And for
            CUDAExecutionProvider, it is recommended to use float32 or float16 models instead of quantized models.

        Args:
            encoder_model_path (str | bytes | PathLike): Path to the encoder model.
            decoder_model_path (str | bytes | PathLike): Path to the decoder model.
            provider (Sequence[str | tuple[str, dict[Any, Any]]], optional): Providers for the model. Defaults to None.
            session_options (ort.SessionOptions, optional): Session options for the model. Defaults to None.
            provider_options (Sequence[dict[Any, Any]], optional): Provider options for the model. Defaults to None.
            use_io_binding (bool, optional): I/O binding for the model. Defaults to None.
            **kwargs: Additional keyword arguments passed to the `onnxruntime.InferenceSession`.

        Raises:
            ValueError: If the encoder and decoder models have different types.
            ValueError: If the dtype is not supported.
            ONNXRuntimeError: Exception from ONNXRuntime.
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
            raise ValueError(err_msg)
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
