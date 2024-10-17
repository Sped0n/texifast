from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps
from tokenizers import Tokenizer

from .config import TxfConfig
from .helpers import logger
from .model import TxfModel

if TYPE_CHECKING:
    from typing import TypeAlias

    from numpy.typing import NDArray

    TxfArray: TypeAlias = NDArray[np.float32 | np.float16]


class TxfPipeline:
    def __init__(
        self,
        model: TxfModel,
        tokenizer: Tokenizer,
        config: TxfConfig | None = None,
    ) -> None:
        # referneces
        self.__model: TxfModel = model
        self.__tokenizer: Tokenizer = tokenizer
        self.__config: TxfConfig
        if config is not None:
            self.__config = config
        else:
            logger.info("No config provided, using default config")
            self.__config = TxfConfig()

        # token ids
        self.__bos_token_id: np.intp = self.__tokenizer.token_to_id(
            self.__config.bos_token
        )
        self.__eos_token_id: np.intp = self.__tokenizer.token_to_id(
            self.__config.eos_token
        )

        # kv_bindings(for I/O binding)
        self.__kv_bindings: list[tuple[str, str]] | None = None

    def __generate(self, pixel_values: TxfArray, max_new_tokens: int) -> list[np.intp]:
        encoder_result: TxfArray = self.__model.encoder_session.run(
            None, {"pixel_values": pixel_values}
        )
        hidden_state: TxfArray = encoder_result[0]

        token_ids: list[np.intp] = [self.__bos_token_id]
        dummy_kv: TxfArray = np.empty((1, 16, 0, 64), dtype=self.__model.dtype)

        decoder_inputs = {
            "encoder_hidden_states": hidden_state,
            "input_ids": np.array([[self.__bos_token_id]], dtype=np.int64),
            "use_cache_branch": np.array([False]),
            **{
                f"past_key_values.{i}.{j}.{k}": dummy_kv
                for i in range(8)
                for j in ["decoder", "encoder"]
                for k in ["key", "value"]
            },
        }

        while len(token_ids) < max_new_tokens:
            decoder_result: list[NDArray[np.float32]] = (
                self.__model.decoder_session.run(None, decoder_inputs)
            )

            new_token_id: np.intp = np.argmax(decoder_result[0][0, -1, :])
            token_ids.append(new_token_id)
            if new_token_id == self.__eos_token_id:
                break
            if len(token_ids) == 2:  # first pass
                decoder_inputs.update(
                    {
                        "input_ids": np.array([[new_token_id]], dtype=np.int64),
                        "use_cache_branch": np.array([True]),
                        **{
                            f"past_key_values.{i}.{j}.{k}": decoder_result[
                                1 + 4 * i + ik if j == "decoder" else 3 + 4 * i + ik
                            ]
                            for i in range(8)
                            for j in ["decoder", "encoder"]
                            for ik, k in enumerate(["key", "value"])
                        },
                    }
                )
            else:
                decoder_inputs.update(
                    {
                        "input_ids": np.array([[new_token_id]], dtype=np.int64),
                        **{
                            f"past_key_values.{i}.decoder.{j}": decoder_result[
                                1 + 4 * i + ij
                            ]
                            for i in range(8)
                            for ij, j in enumerate(["key", "value"])
                        },
                    }
                )

        return token_ids

    def __generate_with_io_binding(
        self, pixel_values: TxfArray, max_new_tokens: int
    ) -> list[np.intp]:
        # create kv bindings if not exists
        if self.__kv_bindings is None:
            self.kv_bindings = [
                (f"past_key_values.{i}.{j}.{k}", f"present.{i}.{j}.{k}")
                for i in range(8)
                for j in ["decoder", "encoder"]
                for k in ["key", "value"]
            ]

        # encoder binding
        encoder_io_binding: ort.IOBinding = self.__model.encoder_session.io_binding()
        encoder_io_binding.bind_cpu_input(
            "pixel_values", np.ascontiguousarray(pixel_values)
        )
        encoder_io_binding.bind_output(
            "last_hidden_state", device_type=self.__model.device_type
        )

        # encoder run
        encoder_io_binding.synchronize_inputs()
        self.__model.encoder_session.run_with_iobinding(encoder_io_binding)
        encoder_io_binding.synchronize_outputs()

        # decoder inputs preparation(first pass/use_cache_branch=False)
        hidden_state_ort_value: ort.OrtValue = encoder_io_binding.get_outputs()[0]
        dummy_kv: TxfArray = np.empty((1, 16, 0, 64), dtype=self.__model.dtype)
        token_ids: list[np.intp] = [self.__bos_token_id]
        input_ids_ort_value: ort.OrtValue = ort.OrtValue.ortvalue_from_numpy(
            np.ascontiguousarray(np.array([[self.__bos_token_id]], dtype=np.int64)),
            device_type="cpu",
        )
        use_cache_ort_value: ort.OrtValue = ort.OrtValue.ortvalue_from_numpy(
            np.ascontiguousarray(np.array([False])),
            device_type=self.__model.device_type,
        )
        dummy_kv_ort_value: ort.OrtValue = ort.OrtValue.ortvalue_from_numpy(
            np.ascontiguousarray(dummy_kv), device_type=self.__model.device_type
        )

        # decoder binding
        decoder_io_binding: ort.IOBinding = self.__model.decoder_session.io_binding()
        decoder_io_binding.bind_ortvalue_input(
            "encoder_hidden_states", hidden_state_ort_value
        )
        decoder_io_binding.bind_ortvalue_input("input_ids", input_ids_ort_value)
        decoder_io_binding.bind_ortvalue_input("use_cache_branch", use_cache_ort_value)
        decoder_io_binding.bind_output(
            "logits", device_type="cpu"
        )  # logits will always be on CPU, beacuse it is numpy that orchestrating behind
        for past, present in self.kv_bindings:
            decoder_io_binding.bind_ortvalue_input(past, dummy_kv_ort_value)
            decoder_io_binding.bind_output(
                present, device_type=self.__model.device_type
            )

        # variables
        ekv_ort_values: dict[str, ort.OrtValue] = {}
        first_pass: bool = True

        while len(token_ids) < max_new_tokens:
            # decoder run
            decoder_io_binding.synchronize_inputs()
            self.__model.decoder_session.run_with_iobinding(decoder_io_binding)
            decoder_io_binding.synchronize_outputs()
            decoder_result: list[ort.OrtValue] = decoder_io_binding.get_outputs()

            # update token_ids
            new_token_id: np.intp = np.argmax(decoder_result[0].numpy()[0, -1, :])
            input_ids_ort_value.update_inplace(
                np.ascontiguousarray(np.array([[new_token_id]], dtype=np.int64))
            )
            token_ids.append(new_token_id)
            if new_token_id == self.__eos_token_id:
                break

            # update use_cache_branch
            if first_pass:
                use_cache_ort_value = ort.OrtValue.ortvalue_from_numpy(  # cannot update inplace an OrtValue if it is in a different device
                    np.ascontiguousarray(np.array([True])),
                    device_type=self.__model.device_type,
                )

            # decoder binding(inputs)
            decoder_io_binding.clear_binding_inputs()
            decoder_io_binding.bind_ortvalue_input(
                "encoder_hidden_states", hidden_state_ort_value
            )
            decoder_io_binding.bind_ortvalue_input("input_ids", input_ids_ort_value)

            decoder_io_binding.bind_ortvalue_input(
                "use_cache_branch", use_cache_ort_value
            )
            for idx, (past, present) in enumerate(self.kv_bindings):
                i, j, k = past.split(".")[1:]
                if j == "encoder":
                    if first_pass:
                        ekv_ort_values[f"{i}.{j}.{k}"] = (
                            ort.OrtValue.ortvalue_from_numpy(  # create a copy here because clear_binding_outputs() will remove the original
                                np.ascontiguousarray(decoder_result[1 + idx].numpy()),
                                device_type=self.__model.device_type,
                            )
                        )
                        first_pass = False
                    decoder_io_binding.bind_ortvalue_input(
                        past, ekv_ort_values[f"{i}.{j}.{k}"]
                    )
                else:
                    decoder_io_binding.bind_ortvalue_input(
                        past, decoder_result[1 + idx]
                    )

            # decoder binding(outputs)
            decoder_io_binding.clear_binding_outputs()
            decoder_io_binding.bind_output(
                "logits", device_type=self.__model.device_type
            )
            for _, present in self.kv_bindings:
                decoder_io_binding.bind_output(
                    present, device_type=self.__model.device_type
                )

        return token_ids

    def __preprocess(self, image: Image.Image) -> TxfArray:
        # RGB
        image = image.convert("RGB")
        # resize
        image.thumbnail(self.__config.size)
        # pad
        delta_width: int = self.__config.size[0] - image.width
        delta_height: int = self.__config.size[1] - image.height
        padding: tuple[int, int, int, int] = (
            delta_width // 2,
            delta_height // 2,
            delta_width - (delta_width // 2),
            delta_height - (delta_height // 2),
        )
        image = ImageOps.expand(image, padding, fill=self.__config.pad_fill_value)
        # rescale
        np_image: TxfArray = (
            np.array(image, dtype=self.__model.dtype) * self.__config.rescale_factor
        )
        # normalize
        mean: TxfArray = np.array(self.__config.image_mean, dtype=self.__model.dtype)
        std: TxfArray = np.array(self.__config.image_std, dtype=self.__model.dtype)
        np_image = (np_image - mean) / std
        # HWC -> CHW -> BCHW
        return np_image.transpose(2, 0, 1).reshape(1, 3, *self.__config.size)

    def __call__(self, image: Image.Image, max_new_tokens: int = 384) -> str:
        pixel_values: TxfArray = self.__preprocess(image)
        token_ids: list[np.intp]
        if self.__model.use_io_binding:
            token_ids = self.__generate_with_io_binding(pixel_values, max_new_tokens)
        else:
            token_ids = self.__generate(pixel_values, max_new_tokens)
        return self.__tokenizer.decode(token_ids)
