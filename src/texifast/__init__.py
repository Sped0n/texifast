from importlib.util import find_spec

from .helpers import logger

ERROR_MSG = """
Onnxruntime not found, please ensure you install texifast with the 'cpu' or 'all' extra
like so:

```
pip install texifast[cpu]
# or if you want to use GPU(cuda) acceleration
pip install texifast[gpu]
```

Directly installing texifast will not install onnxruntime by default!!!

```
pip install texifast # don't do this!!!
```
"""


try:
    assert find_spec("onnxruntime") is not None
except AssertionError:
    logger.error(ERROR_MSG)
    raise ImportError(
        "Onnxruntime not found, please install texifast with the 'cpu' or 'gpu' extra"
    )
