<!-- markdownlint-disable -->

<a href="../src/texifast/config.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `texifast.config`




**Global Variables**
---------------
- **TYPE_CHECKING**
- **IMAGE_STD**
- **IMAGE_MEAN**


---

<a href="../src/texifast/config.py#L14"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TxfConfig`




<a href="../src/texifast/config.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    size: 'tuple[int, int]' = (420, 420),
    pad_fill_value: 'int' = 255,
    rescale_factor: 'float' = 0.00392156862745098,
    image_std: 'list[float]' = [0.229, 0.224, 0.225],
    image_mean: 'list[float]' = [0.485, 0.456, 0.406],
    bos_token: 'str' = '<s>',
    eos_token: 'str' = '</s>',
    decoder_layers: 'int' = 8
) → None
```

Initialize the TxfConfig class. 



**Note:**

> The default values are set based on the `preprocess_config.json` and `config.json` in huggingface `Spedon/texify-quantized-onnx` repository. 
>

**Args:**
 
 - <b>`size`</b> (tuple[int, int], optional):  Size of the input image. Defaults to (420, 420). 
 - <b>`pad_fill_value`</b> (int, optional):  Fill value for padding. Defaults to 255. 
 - <b>`rescale_factor`</b> (float, optional):  Rescale factor for the input image. Defaults to 0.00392156862745098. 
 - <b>`image_std`</b> (list[float], optional):  Standard deviation of the input image. Defaults to IMAGE_STD. 
 - <b>`image_mean`</b> (list[float], optional):  Mean of the input image. Defaults to IMAGE_MEAN. 
 - <b>`bos_token`</b> (str, optional):  Beginning of sentence token. Defaults to "<s>". 
 - <b>`eos_token`</b> (str, optional):  End of sentence token. Defaults to "</s>". 
 - <b>`decoder_layers`</b> (int, optional):  Number of decoder layers. Defaults to 8. 







---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
