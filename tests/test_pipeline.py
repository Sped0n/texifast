from pathlib import Path

from huggingface_hub import hf_hub_download
from PIL import Image
from tokenizers import Tokenizer

from texifast.model import TxfModel
from texifast.pipeline import TxfPipeline

THIS_DIR = Path(__file__).parent


def test_txf_pipeline() -> None:
    revision = open(THIS_DIR.joinpath("model_revision.txt")).read().strip()
    encoder_model_path = THIS_DIR.joinpath("encoder_model_quantized.onnx")
    if not encoder_model_path.exists():
        hf_hub_download(
            "Spedon/texify-quantized-onnx",
            filename="encoder_model_quantized.onnx",
            local_dir=THIS_DIR,
            revision=revision,
        )
    decoder_model_path = THIS_DIR.joinpath("decoder_model_merged_quantized.onnx")
    if not decoder_model_path.exists():
        hf_hub_download(
            "Spedon/texify-quantized-onnx",
            filename="decoder_model_merged_quantized.onnx",
            local_dir=THIS_DIR,
            revision=revision,
        )
    tokenizer_config_path = THIS_DIR.joinpath("tokenizer.json")
    if not tokenizer_config_path.exists():
        hf_hub_download(
            "Spedon/texify-quantized-onnx",
            filename="tokenizer.json",
            local_dir=THIS_DIR,
            revision=revision,
        )
    model = TxfModel(
        encoder_model_path=encoder_model_path,
        decoder_model_path=decoder_model_path,
    )
    tokenizer = Tokenizer.from_file(str(tokenizer_config_path))
    pipeline = TxfPipeline(model=model, tokenizer=tokenizer)
    image_path = Image.open(THIS_DIR.joinpath("latex.png"))
    assert (
        pipeline(image_path)
        == "The potential $V_i$ of cell $\\mathcal{C}_i$ centred at position $\\mathbf{r}_i$ is related to the surface charge densities $\\sigma_j$ of cells $\\mathcal{C}_j$ $j\\in[1,N]$ through the superposition principle as: $$V_i\\,=\\,\\sum_{j=0}^{N}\\,\\frac{\\sigma_j}{4\\pi\\epsilon_0}\\,\\int_{\\mathcal{C}_j}\\frac{1}{\\|\\mathbf{r}_i-\\mathbf{r}^{\\prime}\\|}\\,\\mathrm{d}^2\\mathbf{r}^{\\prime}\\,=\\,\\sum_{j=0}^{N}\\,Q_{ij}\\,\\sigma_j,$$ where the integral over the surface of cell $\\mathcal{C}_j$ only depends on $\\mathcal{C}_j$ shape and on the relative position of the target point $\\mathbf{r}_i$ with respect to $\\mathcal{C}_j$ location, as $\\sigma_j$ is assumed constant over the whole surface of cell $\\mathcal{C}_j$. "
    )
