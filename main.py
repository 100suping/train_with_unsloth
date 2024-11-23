import argparse
import os
from dotenv import load_dotenv
from model import do_it_tuning
from utils import str2bool

def get_config():
    """argparse를 이용해 사용자에게 하이퍼 파라미터를 입력 받는 함수입니다."""

    # parser 생성
    parser = argparse.ArgumentParser(
        prog="Get Hyperparameters",
    )

    parser.add_argument(
        "--project-name",
        default="text2sql",
        type=str,
    )

    parser.add_argument(
        "--run-name",
        default="test",
        type=str,
    )

    parser.add_argument(
        "--dataset-path",
        default="100suping/ko-bird-sql-schema",
        choices=["100suping/ko-bird-sql-schema"],
        type=str,
    )

    parser.add_argument(
        "--model-type",
        default="qwen-2.5",
        choices=["qwen-2.5"],
        type=str,
    )

    parser.add_argument(
        "--model-name",
        default="unsloth/Qwen2.5-Coder-32B-Instruct",
        choices=["unsloth/Qwen2.5-Coder-32B-Instruct"],
        type=str,
    )
    
    parser.add_argument(
        "--quant-bit",
        default=8,
        choices=[4,8],
        type=int,
    )
    
    parser.add_argument(
        "--r",
        default=16,
        choices=[8, 16, 32, 64, 128],
        type=int,
    )
    
    parser.add_argument(
        "--lora-alpha",
        default=16,
        choices=[8, 16, 32, 64, 128],
        type=int,
    )

    parser.add_argument(
        "--lora-dropout",
        default=0,
        type=float,
    )

    parser.add_argument(
        "--max-seq-length",
        default=1400,
        type=int,
    )
    
    parser.add_argument(
        "--output-dir",
        default="outputs",
        type=str,
    )
    
    parser.add_argument(
        "--save-steps",
        default=10,
        type=int,
    )
    
    parser.add_argument(
        "--logging-steps",
        default=2,
        type=int,
    )

    parser.add_argument("--epochs", default=1, type=int)

    parser.add_argument(
        "--batch-size",
        default=1,
        type=int,
    )
    
    parser.add_argument(
        "--warmup-steps",
        default=2,
        type=int,
    )

    parser.add_argument(
        "--lr",
        default=2e-4,
        type=float,
    )
    
    parser.add_argument(
        "--gradient-accumulation-steps",
        default=2,
        type=int,
    )
    
    parser.add_argument(
        "--verbose",
        default=False,
        type=str2bool,
    )

    parser.add_argument(
        "--report-to",
        default="none",
        choices=["none", "wandb"],
        type=str,
    )

    parser.add_argument(
        "--seed",
        default=42,
        type=int,
    )
    
    parser.add_argument(
        "--test-run",
        default=True,
        type=str2bool,
    )

    config = parser.parse_args()

    return config


if __name__ == "__main__":
    config = get_config()

    load_dotenv()
    os.environ["WANDB_PROJECT"] = f"{config.project_name}"
    
    do_it_tuning(config)