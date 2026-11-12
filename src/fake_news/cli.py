"""Command line training and inference entry points."""
import argparse
import json
from pathlib import Path

import joblib

from .pipeline import evaluate, load_dataset, predict_texts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("evaluate", help="train and evaluate on the original Kaggle CSVs")
    train.add_argument("--data", type=Path, default=Path("data/raw"))
    train.add_argument("--output", type=Path, default=Path("results/reproduced"))
    train.add_argument("--model-output", type=Path, default=Path("models/fake_news_pipeline.joblib"))
    predict = commands.add_parser("predict", help="classify one text using a saved pipeline")
    predict.add_argument("text")
    predict.add_argument("--model", type=Path, default=Path("models/fake_news_pipeline.joblib"))
    args = parser.parse_args()
    if args.command == "evaluate":
        result = evaluate(
            load_dataset(args.data), args.output, model_path=args.model_output,
        )
        print(json.dumps(result, indent=2))
    else:
        model = joblib.load(args.model)
        print(json.dumps(predict_texts(model, [args.text])[0], indent=2))


if __name__ == "__main__":
    main()
