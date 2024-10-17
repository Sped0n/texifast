<!-- markdownlint-disable -->

<a href="../src/texifast/pipeline.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `texifast.pipeline`




**Global Variables**
---------------
- **TYPE_CHECKING**


---

<a href="../src/texifast/pipeline.py#L24"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TxfPipeline`




<a href="../src/texifast/pipeline.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    model: 'TxfModel',
    tokenizer: 'Tokenizer | str | Path',
    config: 'TxfConfig | None' = None
) → None
```

Initialize the TxfPipeline class. 



**Args:**
 
 - <b>`model`</b> (TxfModel):  The model to use for the pipeline. 
 - <b>`tokenizer`</b> (Tokenizer | str | Path):  The tokenizer to use for the pipeline. 
 - <b>`config`</b> (TxfConfig, optional):  The configuration to use for the pipeline. Defaults to None. 







---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
