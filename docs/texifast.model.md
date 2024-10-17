<!-- markdownlint-disable -->

<a href="../src/texifast/model.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `texifast.model`




**Global Variables**
---------------
- **TYPE_CHECKING**


---

<a href="../src/texifast/model.py#L16"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TxfModel`




<a href="../src/texifast/model.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    encoder_model_path: 'str | bytes | PathLike[Any]',
    decoder_model_path: 'str | bytes | PathLike[Any]',
    provider: 'Sequence[str | tuple[str, dict[Any, Any]]] | None' = None,
    session_options: 'SessionOptions | None' = None,
    provider_options: 'Sequence[dict[Any, Any]] | None' = None,
    use_io_binding: 'bool | None' = None,
    **kwargs: 'Any'
) → None
```

Initialize the TxfModel class. 



**Note:**

> You should pass in same type(quantized/fp16) of encoder and decoder models, do not mix them. And for CUDAExecutionProvider, it is recommended to use float32 or float16 models instead of quantized models. 
>

**Args:**
 
 - <b>`encoder_model_path`</b> (str | bytes | PathLike):  Path to the encoder model. 
 - <b>`decoder_model_path`</b> (str | bytes | PathLike):  Path to the decoder model. 
 - <b>`provider`</b> (Sequence[str | tuple[str, dict[Any, Any]]], optional):  Providers for the model. Defaults to None. 
 - <b>`session_options`</b> (ort.SessionOptions, optional):  Session options for the model. Defaults to None. 
 - <b>`provider_options`</b> (Sequence[dict[Any, Any]], optional):  Provider options for the model. Defaults to None. 
 - <b>`use_io_binding`</b> (bool, optional):  I/O binding for the model. Defaults to None. 
 - <b>`**kwargs`</b>:  Additional keyword arguments passed to the `onnxruntime.InferenceSession`. 



**Raises:**
 
 - <b>`ValueError`</b>:  If the encoder and decoder models have different types. 
 - <b>`ValueError`</b>:  If the dtype is not supported. 
 - <b>`ONNXRuntimeError`</b>:  Exception from ONNXRuntime. 







---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
