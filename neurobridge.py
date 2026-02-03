from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger

from neurobridge.config import NeuroBridgeConfig
from neurobridge.data_pipeline import PhonemeInventory
from neurobridge.speech import PhonemeSynthesizer
from neurobridge.training import train_and_evaluate


def _load_config(path: Path | str) -> NeuroBridgeConfig:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file {cfg_path} does not exist.")
    logger.info("Loading NeuroBridge configuration from %s", cfg_path)
    return NeuroBridgeConfig.from_yaml(cfg_path)


def cmd_train(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    metrics = train_and_evaluate(config)
    logger.info("Training metrics summary: frame_accuracy=%.4f", metrics["test_results"].get("frame_accuracy", 0.0))
    if args.print_json:
        print(json.dumps(metrics, indent=2))


def cmd_synthesize(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    inventory = PhonemeInventory(config.dataset.phonemes)
    ids = [int(token) for token in args.sequence.split(",") if token.strip()]
    synthesizer = PhonemeSynthesizer(inventory, config.speech)
    audio = synthesizer.synthesize(ids)
    if audio.size == 0:
        logger.warning("No audio generated because the phoneme sequence was empty.")
        return
    destination = Path(args.output or (config.speech.export_audio_dir / "sample.wav"))
    synthesizer.save_wav(audio, destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NeuroBridge CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and evaluate the offline decoder.")
    train_parser.add_argument("--config", type=Path, default=Path("neurobridge.config.yaml"), help="Path to YAML config.")
    train_parser.add_argument("--print-json", action="store_true", help="Echo metrics JSON to stdout.")
    train_parser.set_defaults(func=cmd_train)

    synth_parser = subparsers.add_parser("synthesize", help="Synthesize audio from a phoneme id sequence.")
    synth_parser.add_argument("--config", type=Path, default=Path("neurobridge.config.yaml"))
    synth_parser.add_argument(
        "--sequence",
        type=str,
        required=True,
        help="Comma separated phoneme id sequence, e.g. '0,1,2,3'.",
    )
    synth_parser.add_argument("--output", type=Path, help="Optional output WAV path.")
    synth_parser.set_defaults(func=cmd_synthesize)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
