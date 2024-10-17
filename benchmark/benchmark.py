import argparse
from pathlib import Path
from time import time

from huggingface_hub import hf_hub_download

from texifast.model import TxfModel
from texifast.pipeline import TxfPipeline

DATA_DIR = Path(__file__).parent.parent.joinpath("tests")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Texifast benchmark")
    parser.add_argument("--cuda", action="store_true", help="Run benchmark on CUDA")
    parser.add_argument(
        "--iob", action="store_true", help="Run benchmark with I/O binding"
    )
    parser.add_argument(
        "--optimum",
        action="store_true",
        help="Run same benchmark with huggingface optimum, need to install optimum manually",
    )
    parser.add_argument("-r", "--rounds", type=int, default=10, help="Number of rounds")
    args = parser.parse_args()

    revision = open(DATA_DIR.joinpath("model_revision.txt")).read().strip()

    if args.optimum:
        from optimum.onnxruntime import ORTModelForVision2Seq
        from optimum.pipelines import pipeline

        model = ORTModelForVision2Seq.from_pretrained(
            "Spedon/texify-quantized-onnx",
            revision=revision,
            provider="CUDAExecutionProvider" if args.cuda else "CPUExecutionProvider",
            use_io_binding=args.iob,
        )
        benchmark_pipeline = pipeline(
            "image-to-text",
            model,
            feature_extractor="Spedon/texify-quantized-onnx",
            image_processor="Spedon/texify-quantized-onnx",
        )

    else:
        encoder_model_path = DATA_DIR.joinpath("encoder_model_quantized.onnx")
        if not encoder_model_path.exists():
            hf_hub_download(
                "Spedon/texify-quantized-onnx",
                filename="encoder_model_quantized.onnx",
                local_dir=DATA_DIR,
                revision=revision,
            )
        decoder_model_path = DATA_DIR.joinpath("decoder_model_merged_quantized.onnx")
        if not decoder_model_path.exists():
            hf_hub_download(
                "Spedon/texify-quantized-onnx",
                filename="decoder_model_merged_quantized.onnx",
                local_dir=DATA_DIR,
                revision=revision,
            )
        tokenizer_json_path = DATA_DIR.joinpath("tokenizer.json")
        if not tokenizer_json_path.exists():
            hf_hub_download(
                "Spedon/texify-quantized-onnx",
                filename="tokenizer.json",
                local_dir=DATA_DIR,
                revision=revision,
            )
        model = TxfModel(
            encoder_model_path=encoder_model_path,
            decoder_model_path=decoder_model_path,
            provider=["CUDAExecutionProvider"]
            if args.cuda
            else ["CPUExecutionProvider"],
            use_io_binding=args.iob,
        )
        benchmark_pipeline = TxfPipeline(model=model, tokenizer=tokenizer_json_path)

    image_path = str(DATA_DIR.joinpath("latex.png"))
    avg_duration = None
    for _ in range(args.rounds):
        print(f"Round {_ + 1}/{args.rounds}", end="\r")
        start = time()
        benchmark_pipeline(image_path, max_new_tokens=384)
        duration = time() - start
        avg_duration = (
            duration if avg_duration is None else (avg_duration + duration) / 2
        )
    print(f"Average duration: {avg_duration:.3f} seconds")
