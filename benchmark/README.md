# Benchmark

> The benchmark results are generated using `benchmark.py` and [memray](https://github.com/bloomberg/memray).
>
> All memray result can be found in the `memray` folder.

## macOS

### Hardware Specs

- CPU: Apple M1 Pro (6P2E)

- Memory: 16GB

### Environment

```
❯ python --version && uv pip list
Python 3.11.10
Package            Version     Editable project location
------------------ ----------- ----------------------------------
aiohappyeyeballs   2.4.3
aiohttp            3.10.10
aiosignal          1.3.1
attrs              24.2.0
certifi            2024.8.30
charset-normalizer 3.4.0
click              8.1.7
coloredlogs        15.0.1
datasets           3.0.1
dill               0.3.8
evaluate           0.4.3
filelock           3.16.1
flatbuffers        24.3.25
frozenlist         1.4.1
fsspec             2024.6.1
huggingface-hub    0.25.2
humanfriendly      10.0
idna               3.10
iniconfig          2.0.0
jinja2             3.1.4
lazydocs           0.4.8
linkify-it-py      2.0.3
markdown-it-py     3.0.0
markupsafe         3.0.1
mdit-py-plugins    0.4.2
mdurl              0.1.2
memray             1.14.0
mpmath             1.3.0
multidict          6.1.0
multiprocess       0.70.16
networkx           3.4.1
numpy              2.0.2
onnx               1.17.0
onnxruntime        1.19.2
optimum            1.23.1
packaging          24.1
pandas             2.2.3
pillow             11.0.0
pip                24.0
platformdirs       4.3.6
pluggy             1.5.0
propcache          0.2.0
protobuf           5.28.2
pyarrow            17.0.0
pygments           2.18.0
pytest             8.3.3
python-dateutil    2.9.0.post0
pytz               2024.2
pyyaml             6.0.2
regex              2024.9.11
requests           2.32.3
rich               13.9.2
safetensors        0.4.5
sentencepiece      0.2.0
setuptools         65.5.0
shellingham        1.5.4
six                1.16.0
sympy              1.13.3
texifast           0.1.0       /Users/spedon/eden/python/texifast
textual            0.83.0
tokenizers         0.20.1
torch              2.4.1
tqdm               4.66.5
transformers       4.45.2
typer              0.12.5
typing-extensions  4.12.2
tzdata             2024.2
uc-micro-py        1.0.3
urllib3            2.2.3
wheel              0.44.0
xxhash             3.5.0
yarl               1.15.4
```

### Result

|                                                                 | texifast(with io_binding) | texifast(no io_binding) | Optimum(with io_binding) | Optimum(no io_binding) |
| :-------------------------------------------------------------: | :-----------------------: | :---------------------: | :----------------------: | :--------------------: |
|               **Avg Inference time of 10 rounds**               |          1.668s           |         1.612s          |          1.915s          |         2.447s         |
| **Memory usage during inference**(_profile result from memray_) |            ~1G            |           ~1G           |          ~1.2G           |         ~1.2G          |
