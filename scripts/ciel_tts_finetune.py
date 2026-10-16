"""Prepare verified Ciel datasets and experimental pretrained-model adaptation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    from scripts.tts_finetune.common import ROOT

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("directory", type=Path)
    preparation.add_argument("--source", type=Path, default=ROOT / "data/voice/source/ciel")
    asr = commands.add_parser("transcribe")
    asr.add_argument("directory", type=Path)
    asr.add_argument("--model", choices=["small.en", "medium.en"], default="medium.en")
    asr.add_argument("--reuse-from", type=Path, help="Reuse only identical audio hashes/model")
    review = commands.add_parser("review", help="Render per-clip local verification page")
    review.add_argument("directory", type=Path)
    imports = commands.add_parser("import-reviews")
    imports.add_argument("directory", type=Path)
    imports.add_argument("reviews", type=Path)
    export = commands.add_parser("export", help="Export at least 30 explicitly verified minutes")
    export.add_argument("directory", type=Path)
    export.add_argument("output", type=Path)
    assets = commands.add_parser("xtts-assets", help="Download pinned pretrained training assets")
    assets.add_argument("directory", type=Path)
    training = commands.add_parser("train-xtts", help="Native GPT pilot; rejects a 6 GB GPU")
    training.add_argument("dataset", type=Path)
    training.add_argument("models", type=Path)
    training.add_argument("output", type=Path)
    training.add_argument("--epochs", type=int, default=2)
    training.add_argument("--restore", type=Path, help="Strict full-state recovery checkpoint")
    training.add_argument("--resume", action="store_true", help="Resume committed latest state")
    sanity = commands.add_parser("cloud-check", help="Validate dataset/tokenizer, no training")
    sanity.add_argument("dataset", type=Path)
    sanity.add_argument("models", type=Path)
    sanity.add_argument("--baseline", type=Path)
    final = commands.add_parser("export-cloud", help="Package best inference or full recovery")
    final.add_argument("workspace", type=Path)
    final.add_argument("models", type=Path)
    final.add_argument("dataset", type=Path)
    final.add_argument("output", type=Path)
    final.add_argument("--recovery", action="store_true")
    register = commands.add_parser(
        "import-xtts", help="Register experimental bundle, no activation")
    register.add_argument("archive", type=Path)
    register.add_argument("directory", type=Path)
    baseline = commands.add_parser("cloud-baseline", help="Reuse accepted Chatterbox audio first")
    baseline.add_argument("directory", type=Path)
    inference = commands.add_parser("inference-xtts", help="Export weights without optimizer state")
    inference.add_argument("checkpoint", type=Path)
    inference.add_argument("output", type=Path)
    bundle = commands.add_parser("bundle", help="Prepare private code/dataset ZIPs; no upload")
    bundle.add_argument("directory", type=Path)
    bundle.add_argument("--dataset", type=Path)
    bundle.add_argument("--baseline", type=Path)
    notebook = commands.add_parser("notebook")
    notebook.add_argument("path", type=Path)
    evaluation = commands.add_parser("evaluate", help="Three warm trials; no production changes")
    evaluation.add_argument("engine", choices=["xtts", "chatterbox", "styletts2"])
    evaluation.add_argument("directory", type=Path)
    evaluation.add_argument("--checkpoint", type=Path)
    evaluation.add_argument("--experiment", type=Path)
    evaluation.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    evaluation.add_argument("--iterations", type=int, default=3)
    evaluation.add_argument("--sustained-seconds", type=int, default=0)
    listening = commands.add_parser("listening", help="Build rating-gated checkpoint comparison")
    listening.add_argument("directory", type=Path)
    listening.add_argument("--results", nargs="*", type=Path, default=[])
    args = parser.parse_args()
    if args.command == "prepare":
        from scripts.tts_finetune.corpus import prepare

        print(prepare(args.source, args.directory))
    elif args.command == "transcribe":
        from scripts.tts_finetune.corpus import transcribe

        print(transcribe(args.directory, args.model, args.reuse_from))
    elif args.command == "review":
        from scripts.tts_finetune.review import render

        print(render(args.directory.resolve()))
    elif args.command == "import-reviews":
        from scripts.tts_finetune.review import import_reviews

        print(import_reviews(args.directory.resolve(), args.reviews))
    elif args.command == "export":
        from scripts.tts_finetune.review import export

        print(export(args.directory.resolve(), args.output.resolve()))
    elif args.command == "xtts-assets":
        from scripts.tts_finetune.xtts import assets

        print(assets(args.directory.resolve()))
    elif args.command == "train-xtts":
        from scripts.tts_finetune.xtts import train

        print(train(args.dataset.resolve(), args.models.resolve(), args.output.resolve(),
                    args.epochs, args.restore.resolve() if args.restore else None, args.resume))
    elif args.command == "cloud-check":
        from scripts.tts_finetune.training import dataset_report

        dataset_report(args.dataset.resolve(), args.models.resolve())
        if args.baseline:
            from scripts.tts_finetune.cloud import validate_baseline

            validate_baseline(args.baseline.resolve(), args.dataset.resolve())
    elif args.command == "export-cloud":
        from scripts.tts_finetune.artifacts import export_cloud

        print(export_cloud(args.workspace.resolve(), args.models.resolve(), args.dataset.resolve(),
                           args.output.resolve(), args.recovery))
    elif args.command == "import-xtts":
        from scripts.tts_finetune.artifacts import import_experiment

        print(import_experiment(args.archive.resolve(), args.directory.resolve()))
    elif args.command == "cloud-baseline":
        from scripts.tts_finetune.cloud import prepare_baseline

        print(prepare_baseline(args.directory.resolve()))
    elif args.command == "bundle":
        from scripts.tts_finetune.cloud import bundle

        print(bundle(args.directory.resolve(), args.dataset.resolve() if args.dataset else None,
                     args.baseline.resolve() if args.baseline else None))
    elif args.command == "inference-xtts":
        from scripts.tts_finetune.xtts import inference_checkpoint

        print(inference_checkpoint(args.checkpoint.resolve(), args.output.resolve()))
    elif args.command == "notebook":
        from scripts.tts_finetune.cloud import notebook

        print(notebook(args.path.resolve()))
    elif args.command == "evaluate":
        from scripts.tts_finetune.evaluate import evaluate

        print(evaluate(args.engine, args.directory.resolve(),
                       args.checkpoint.resolve() if args.checkpoint else None,
                       args.device, args.iterations, args.sustained_seconds,
                       args.experiment.resolve() if args.experiment else None))
    elif args.command == "listening":
        from scripts.tts_finetune.evaluate import listening

        print(listening(args.directory.resolve(), args.results))


if __name__ == "__main__":
    main()
