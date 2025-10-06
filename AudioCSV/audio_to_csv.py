#!/usr/bin/env python3
"""Audio to CSV feature extraction tool.

This script converts MP3 (and other supported audio formats) into a tabular
representation (CSV/Parquet/JSON) containing per-window audio descriptors such
as pitch, amplitude (RMS), loudness, spectral descriptors, and more.

The tool can be used in two modes:
  * Command line mode – pass arguments directly via flags.
  * Guided mode – run the script without arguments and answer the prompts.

Examples
--------
1. Guided mode (asks you a few questions and runs with sensible defaults):

    python audio_to_csv.py

2. Command line mode with CSV output every 0.25 seconds including pitch and RMS:

    python audio_to_csv.py --input song.mp3 --output song.csv --window 0.25 \
        --features rms,pitch

3. Process every MP3 in a folder, keeping the default feature set and writing
   metadata alongside each CSV:

    python audio_to_csv.py --glob "music/*.mp3" --format csv --write_metadata

The default feature set is ``rms,pitch,lufs,flux,rolloff,dynamic_range,crest_factor``.
You can request additional descriptors with the ``--features`` option.

Supported feature names:
    rms, peak, pitch, voiced_prob, voiced_flag, lufs, flux, rolloff,
    dynamic_range, crest_factor, chroma, tempo, beat_index

See the README in the AudioCSV directory for an end-to-end walkthrough.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import librosa
    import librosa.display  # noqa: F401 (ensures librosa's optional registries load)
except ImportError as exc:  # pragma: no cover - user guidance
    raise SystemExit(
        "librosa is required for audio_to_csv.py. Install it with 'pip install librosa'."
    ) from exc

try:
    import pyloudnorm
except ImportError:  # pragma: no cover - user guidance
    pyloudnorm = None


CHROMA_LABELS = [
    "chroma_C",
    "chroma_C#",
    "chroma_D",
    "chroma_D#",
    "chroma_E",
    "chroma_F",
    "chroma_F#",
    "chroma_G",
    "chroma_G#",
    "chroma_A",
    "chroma_A#",
    "chroma_B",
]


@dataclass
class FeatureConfig:
    features: Sequence[str]
    window: float
    hop: float
    fmin: float
    fmax: float
    ema: Optional[float]
    trim_db: Optional[float]
    pitch_guard_z: Optional[float]
    silence_threshold_db: Optional[float]
    write_metadata: bool
    format: str
    metadata_name: str


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert audio files into feature tables (CSV/JSON/Parquet).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, help="Path to an audio file to analyse")
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination path for the table. Auto-generated if omitted.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        help="Process all files that match the glob pattern (e.g. 'audio/*.mp3').",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=0.25,
        help="Window size in seconds for feature aggregation.",
    )
    parser.add_argument(
        "--hop",
        type=float,
        help="Hop size in seconds. Defaults to the window size.",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="rms,pitch,lufs,flux,rolloff,dynamic_range,crest_factor",
        help="Comma-separated list of features to compute.",
    )
    parser.add_argument(
        "--ema",
        type=float,
        help="Apply exponential moving average smoothing with the provided alpha.",
    )
    parser.add_argument(
        "--trim_db",
        type=float,
        help="Trim leading/trailing sections below this dB threshold (librosa.effects.trim).",
    )
    parser.add_argument(
        "--silence_threshold_db",
        type=float,
        default=-50.0,
        help="Mark frames quieter than this RMS (dB) as silence.",
    )
    parser.add_argument(
        "--fmin",
        type=float,
        default=65.0,
        help="Minimum frequency (Hz) searched by the pitch tracker.",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=1000.0,
        help="Maximum frequency (Hz) searched by the pitch tracker.",
    )
    parser.add_argument(
        "--pitch_guard_z",
        type=float,
        default=3.5,
        help="Robust z-score threshold used to drop pitch outliers.",
    )
    parser.add_argument(
        "--format",
        choices=("csv", "json", "parquet"),
        default="csv",
        help="Output format.",
    )
    parser.add_argument(
        "--write_metadata",
        action="store_true",
        help="Write a JSON sidecar with analysis parameters.",
    )
    parser.add_argument(
        "--metadata_name",
        type=str,
        default="features.meta.json",
        help="Metadata filename written next to the output file when --write_metadata is set.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Force guided mode even if arguments are supplied.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist.",
    )
    args = parser.parse_args(argv)

    if not args.input and not args.glob and not args.interactive:
        # Enter guided mode if nothing was provided.
        args.interactive = True

    return args


def prompt_for_missing_args(args: argparse.Namespace) -> argparse.Namespace:
    """Interactively gather settings from the user when needed."""

    def ask_path(prompt: str, default: Optional[Path] = None) -> Path:
        while True:
            raw = input(f"{prompt}{f' [{default}]' if default else ''}: ").strip()
            if not raw and default is not None:
                return default
            path = Path(raw)
            if path.exists() or path.parent.exists():
                return path
            print("❗ Path does not exist – please try again.")

    if not args.input and not args.glob:
        choice = input(
            "Do you want to analyse a single file or a folder? (file/folder) [file]: "
        ).strip() or "file"
        if choice.lower().startswith("f") and choice.lower() != "file":
            args.glob = input(
                "Enter a glob pattern (e.g. 'audio/*.mp3' or '**/*.wav'): "
            ).strip()
        else:
            args.input = ask_path("Path to audio file")

    if args.input and not args.output:
        suggested = args.input.with_suffix(".csv")
        args.output = ask_path("Where should the table be saved?", suggested)

    if not args.window:
        raw = input("Window size in seconds [0.25]: ").strip()
        args.window = float(raw or 0.25)

    if not args.features:
        raw = input(
            "Features to compute (comma separated) [rms,pitch,lufs,flux,rolloff]: "
        ).strip()
        args.features = raw or "rms,pitch,lufs,flux,rolloff"

    overwrite = input("Overwrite existing files if needed? (y/N): ").strip().lower()
    args.overwrite = overwrite.startswith("y")

    return args


def resolve_files(args: argparse.Namespace) -> List[Tuple[Path, Path]]:
    """Return list of (input_path, output_path) pairs based on CLI flags."""
    pairs: List[Tuple[Path, Path]] = []
    if args.glob:
        root = Path.cwd()
        for path in sorted(root.glob(args.glob)):
            if not path.is_file():
                continue
            output = args.output
            if output and (output.is_dir() or not output.suffix):
                out_path = output / f"{path.stem}.{args.format}"
            elif output:
                out_path = output
            else:
                out_path = path.with_suffix(f".{args.format}")
            pairs.append((path, out_path))
    elif args.input:
        input_path = Path(args.input)
        if args.output:
            output_path = Path(args.output)
            if output_path.is_dir() or not output_path.suffix:
                output_path = output_path / f"{input_path.stem}.{args.format}"
        else:
            output_path = input_path.with_suffix(f".{args.format}")
        pairs.append((input_path, output_path))
    return pairs


def ensure_can_write(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists. Use --overwrite to replace it or change the output path."
        )
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def prepare_feature_config(args: argparse.Namespace) -> FeatureConfig:
    feature_list = [f.strip() for f in args.features.split(",") if f.strip()]
    return FeatureConfig(
        features=feature_list,
        window=float(args.window),
        hop=float(args.hop) if args.hop else float(args.window),
        fmin=float(args.fmin),
        fmax=float(args.fmax),
        ema=float(args.ema) if args.ema is not None else None,
        trim_db=float(args.trim_db) if args.trim_db is not None else None,
        pitch_guard_z=float(args.pitch_guard_z) if args.pitch_guard_z is not None else None,
        silence_threshold_db=float(args.silence_threshold_db)
        if args.silence_threshold_db is not None
        else None,
        write_metadata=bool(args.write_metadata),
        format=args.format,
        metadata_name=args.metadata_name,
    )


def load_audio(path: Path, trim_db: Optional[float]) -> Tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=None, mono=True)
    if trim_db is not None:
        y, _ = librosa.effects.trim(y, top_db=trim_db)
    return y, sr


def frame_parameters(sr: int, window: float, hop: float) -> Tuple[int, int]:
    frame_length = max(1, int(round(window * sr)))
    hop_length = max(1, int(round(hop * sr)))
    if hop_length > frame_length:
        hop_length = frame_length
    return frame_length, hop_length


def compute_rms_and_peak(y: np.ndarray, frame_length: int, hop_length: int) -> Tuple[np.ndarray, np.ndarray]:
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    pad_width = int(frame_length // 2)
    padded = np.pad(y, (pad_width, pad_width), mode="reflect")
    frames = librosa.util.frame(padded, frame_length=frame_length, hop_length=hop_length)
    peak = np.max(np.abs(frames), axis=0)
    return rms, peak


def compute_pitch(
    y: np.ndarray,
    sr: int,
    frame_length: int,
    hop_length: int,
    fmin: float,
    fmax: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        pitch, voiced_flag, voiced_prob = librosa.pyin(
            y,
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length,
            fmin=fmin,
            fmax=fmax,
        )
    except Exception:
        pitch = np.full((math.ceil(len(y) / hop_length)), np.nan)
        voiced_flag = np.zeros_like(pitch, dtype=bool)
        voiced_prob = np.zeros_like(pitch)
    return pitch, voiced_flag.astype(int), voiced_prob


def robust_zscore(values: np.ndarray) -> np.ndarray:
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    if mad == 0 or np.isnan(mad):
        return np.zeros_like(values)
    return 0.6745 * (values - median) / mad


def compute_loudness(y: np.ndarray, sr: int, frame_length: int, hop_length: int) -> np.ndarray:
    if pyloudnorm is None:
        raise RuntimeError(
            "pyloudnorm is required for LUFS computation. Install it with 'pip install pyloudnorm'."
        )
    meter = pyloudnorm.Meter(sr)
    # Segment-wise loudness
    pad_width = int(frame_length // 2)
    padded = np.pad(y, (pad_width, pad_width), mode="reflect")
    frames = librosa.util.frame(padded, frame_length=frame_length, hop_length=hop_length)
    loudness = np.array([meter.measure_loudness(frame.astype(float)) for frame in frames.T])
    return loudness


def compute_flux(y: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    return onset_env


def compute_rolloff(y: np.ndarray, sr: int, frame_length: int, hop_length: int) -> np.ndarray:
    rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr, roll_percent=0.85, hop_length=hop_length, n_fft=frame_length
    )[0]
    return rolloff


def compute_dynamic_range_and_crest(rms: np.ndarray, peak: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    eps = 1e-10
    crest = peak / (rms + eps)
    crest_db = 20 * np.log10(np.maximum(crest, eps))
    dynamic_range_db = 20 * np.log10(np.maximum(peak, eps)) - 20 * np.log10(np.maximum(rms, eps))
    return dynamic_range_db, crest_db


def compute_chroma(y: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    return chroma


def compute_tempo_and_beats(y: np.ndarray, sr: int, hop_length: int) -> Tuple[float, np.ndarray]:
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    return tempo, beat_times


def build_dataframe(
    y: np.ndarray,
    sr: int,
    config: FeatureConfig,
) -> pd.DataFrame:
    frame_length, hop_length = frame_parameters(sr, config.window, config.hop)
    approx_frames = max(1, int(np.ceil(len(y) / hop_length)))
    times = librosa.frames_to_time(
        np.arange(approx_frames), sr=sr, hop_length=hop_length
    )

    features: dict[str, np.ndarray] = {}

    if "rms" in config.features or "peak" in config.features or "dynamic_range" in config.features or "crest_factor" in config.features:
        rms, peak = compute_rms_and_peak(y, frame_length, hop_length)
        if "rms" in config.features:
            features["rms"] = rms
        if "peak" in config.features:
            features["peak"] = peak
        if "dynamic_range" in config.features or "crest_factor" in config.features:
            dynamic_range_db, crest_db = compute_dynamic_range_and_crest(rms, peak)
            if "dynamic_range" in config.features:
                features["dynamic_range"] = dynamic_range_db
            if "crest_factor" in config.features:
                features["crest_factor"] = crest_db

    if "pitch" in config.features or "voiced_prob" in config.features or "voiced_flag" in config.features:
        pitch, voiced_flag, voiced_prob = compute_pitch(
            y, sr, frame_length, hop_length, config.fmin, config.fmax
        )
        if config.pitch_guard_z is not None:
            z = robust_zscore(pitch)
            pitch = np.where(np.abs(z) > config.pitch_guard_z, np.nan, pitch)
        if "pitch" in config.features:
            features["pitch"] = pitch
        if "voiced_flag" in config.features:
            features["voiced_flag"] = voiced_flag
        if "voiced_prob" in config.features:
            features["voiced_prob"] = voiced_prob

    if "lufs" in config.features:
        loudness = compute_loudness(y, sr, frame_length, hop_length)
        features["lufs"] = loudness

    if "flux" in config.features:
        flux = compute_flux(y, sr, hop_length)
        features["flux"] = flux

    if "rolloff" in config.features:
        rolloff = compute_rolloff(y, sr, frame_length, hop_length)
        features["rolloff"] = rolloff

    if "chroma" in config.features:
        chroma = compute_chroma(y, sr, hop_length)
        for label, values in zip(CHROMA_LABELS, chroma):
            features[label] = values

    if "tempo" in config.features or "beat_index" in config.features:
        tempo, beat_times = compute_tempo_and_beats(y, sr, hop_length)
        frame_hint = min((len(arr) for arr in features.values()), default=len(times))
        if "tempo" in config.features:
            features["tempo"] = np.full(frame_hint, tempo)
        if "beat_index" in config.features:
            beat_index = map_beats_to_frames(
                beat_times,
                config.window,
                hop_length,
                sr,
                frame_hint,
            )
            features["beat_index"] = beat_index

    frame_count = min((len(arr) for arr in features.values()), default=len(times))
    times = times[:frame_count]
    start_times = np.maximum(times - config.window / 2.0, 0.0)
    end_times = start_times + config.window

    data = {
        "time_start": start_times,
        "time_end": end_times,
    }
    for key, values in features.items():
        data[key] = values[:frame_count]

    df = pd.DataFrame(data)

    if config.silence_threshold_db is not None and "rms" in data:
        rms = np.asarray(data["rms"])
        rms_db = 20 * np.log10(np.maximum(rms, 1e-12))
        df["is_silence"] = rms_db < config.silence_threshold_db

    if config.ema:
        alpha = config.ema
        for column in df.columns:
            if column in {"time_start", "time_end"}:
                continue
            if pd.api.types.is_numeric_dtype(df[column]):
                df[column] = df[column].ewm(alpha=alpha).mean()

    return df


def map_beats_to_frames(
    beat_times: np.ndarray,
    window: float,
    hop_length: int,
    sr: int,
    length_hint: int,
) -> np.ndarray:
    if length_hint <= 0:
        return np.array([])
    hop_seconds = hop_length / sr
    indices = np.arange(length_hint)
    start = np.maximum(indices * hop_seconds - window / 2.0, 0.0)
    end = start + window
    beat_index = np.zeros(length_hint)
    for idx, (s, e) in enumerate(zip(start, end)):
        hits = np.where((beat_times >= s) & (beat_times < e))[0]
        beat_index[idx] = hits[0] + 1 if hits.size else 0
    return beat_index


def write_table(df: pd.DataFrame, path: Path, format_: str) -> None:
    if format_ == "csv":
        df.to_csv(path, index=False)
    elif format_ == "json":
        df.to_json(path, orient="records", lines=True)
    elif format_ == "parquet":
        try:
            df.to_parquet(path, index=False)
        except ImportError as exc:  # pragma: no cover - guidance
            raise SystemExit(
                "pyarrow or fastparquet is required for Parquet export. Install with 'pip install pyarrow'."
            ) from exc
    else:  # pragma: no cover - safety net
        raise ValueError(f"Unsupported format: {format_}")


def write_metadata(metadata_path: Path, df: pd.DataFrame, config: FeatureConfig, source: Path) -> None:
    payload = {
        "source_file": str(source),
        "sample_rate": df.attrs.get("sample_rate"),
        "window_seconds": config.window,
        "hop_seconds": config.hop,
        "features": list(config.features),
        "ema": config.ema,
        "trim_db": config.trim_db,
        "silence_threshold_db": config.silence_threshold_db,
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "row_count": len(df),
        "format": config.format,
    }
    metadata_path.write_text(json.dumps(payload, indent=2))


def analyse_file(input_path: Path, output_path: Path, config: FeatureConfig, overwrite: bool) -> Path:
    ensure_can_write(output_path, overwrite)
    y, sr = load_audio(input_path, config.trim_db)
    df = build_dataframe(y, sr, config)
    df.attrs["sample_rate"] = sr
    write_table(df, output_path, config.format)
    if config.write_metadata:
        metadata_path = output_path.with_name(config.metadata_name)
        if metadata_path.exists() and not overwrite:
            raise FileExistsError(
                f"Metadata file {metadata_path} already exists. Use --overwrite or change --metadata_name."
            )
        write_metadata(metadata_path, df, config, input_path)
    return output_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.interactive:
        print("🟡 Guided mode: answer a couple of questions to get started. Press Ctrl+C to exit.")
        args = prompt_for_missing_args(args)

    config = prepare_feature_config(args)
    file_pairs = resolve_files(args)
    if not file_pairs:
        raise SystemExit("No input files were found. Double-check --input or --glob.")

    for input_path, output_path in file_pairs:
        print(f"Analyzing {input_path} → {output_path}")
        try:
            result = analyse_file(input_path, output_path, config, args.overwrite)
            print(f"  ✔ Saved {result}")
        except Exception as exc:
            print(f"  ✖ Failed to process {input_path}: {exc}")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    sys.exit(main())
