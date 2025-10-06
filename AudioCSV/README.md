# AudioCSV – MP3 Feature Extractor

AudioCSV is a beginner-friendly Python utility that turns an audio file (MP3, WAV,
FLAC, …) into a spreadsheet-friendly table of musical and production features such
as pitch, loudness, spectral flux, and chroma. You can run it interactively with
no prior experience or automate it with command line flags when you are ready.

> **Need to download the ready-to-run files?**
> * [`audio_to_csv.py`](https://raw.githubusercontent.com/CmouseY/CmouseY/main/AudioCSV/audio_to_csv.py)
> * [`run_audio_to_csv.bat`](https://raw.githubusercontent.com/CmouseY/CmouseY/main/AudioCSV/run_audio_to_csv.bat)
>
> Save both files inside the same folder (for example `AudioCSV/`) and follow the
> quick-start steps below.

---

## 1. Requirements (install once)

1. Install [Python 3.9+](https://www.python.org/downloads/) if you do not already
   have it. During installation on Windows, tick the option **“Add Python to PATH”**.
2. Open a terminal:
   * **Windows:** press <kbd>Win</kbd> + <kbd>R</kbd>, type `cmd`, and press Enter.
   * **macOS:** open **Terminal** from Applications → Utilities.
   * **Linux:** open your favourite terminal emulator.
3. Install the Python packages AudioCSV uses:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install librosa pyloudnorm pandas numpy soundfile pyarrow
   ```

   *`pyarrow` is only required when exporting Parquet files; it is safe to install
   it even if you only use CSV.*

---

## 2. Guided one-click run (no command line knowledge required)

1. Download `audio_to_csv.py` and `run_audio_to_csv.bat` to the same folder.
2. Double-click `run_audio_to_csv.bat`.
3. Answer the on-screen questions:
   * choose a file or a folder of MP3s,
   * pick where the CSV should be saved,
   * accept the default feature set or type your own.
4. The script prints progress (✔ success, ✖ error). When it finishes you will find
   a CSV file next to your audio with columns such as `time_start`, `time_end`,
   `rms`, `pitch`, `lufs`, etc.

---

## 3. Command line quick start

```bash
# Analyse a single file every 0.25 s and write CSV + metadata JSON
python audio_to_csv.py --input song.mp3 --output song.csv --window 0.25 \
    --features rms,pitch,lufs,flux,rolloff,dynamic_range,crest_factor \
    --write_metadata --overwrite

# Process a whole folder (writes a CSV per file)
python audio_to_csv.py --glob "music/**/*.mp3" --format csv --overwrite
```

Run `python audio_to_csv.py --help` to see all options.

---

## 4. Feature glossary & ideas

AudioCSV is ready for experimentation. Mix and match descriptors with the
`--features` flag:

| Feature | Description |
| --- | --- |
| `rms` | Root-mean-square energy (amplitude proxy). |
| `peak` | Highest absolute amplitude per window. |
| `pitch` | Fundamental frequency (Hz) estimated with `librosa.pyin`. |
| `voiced_prob`, `voiced_flag` | Confidence from the pitch tracker to tame octave jumps. |
| `lufs` | Loudness in LUFS using `pyloudnorm` – closer to human perception. |
| `flux` | Spectral flux, useful for detecting onsets/motion. |
| `rolloff` | 85% spectral rolloff (timbre/brightness). |
| `dynamic_range`, `crest_factor` | Punch/transient metrics derived from RMS and peak. |
| `chroma` | Twelve chroma bins (`chroma_C`…`chroma_B`) hinting at harmony/key. |
| `tempo`, `beat_index` | Global tempo estimate and beat hits per window. |

Advanced controls exposed via flags:

* **Smoothing:** `--ema 0.2` applies an exponential moving average to smooth
  jittery values.
* **Silence handling:** `--trim_db -40` trims quiet ends, while
  `--silence_threshold_db -45` labels quiet frames with `is_silence = True`.
* **Octave guard:** `--pitch_guard_z 3.0` rejects pitch outliers via robust
  z-score thresholding.
* **Alternate formats:** `--format parquet` or `--format json` for other tools.
* **Batch mode:** `--glob "*.mp3"` analyses multiple files and automatically names
  each output. Combine with `--write_metadata` to produce a companion
  `features.meta.json` per track.

Future enhancements you could explore:

* Streaming/online processing for hour-long recordings.
* Optional harmonic-percussive source separation before pitch tracking.
* Multi-process batch execution for large libraries.
* Beat-synchronised exports (one row per beat instead of fixed windows).

---

## 5. Troubleshooting

* **Missing package errors** – rerun the `pip install` line above. On macOS/Linux
  you may need `python3` instead of `python`.
* **`pyloudnorm` import error** – install it or drop `lufs` from `--features`.
* **`pyarrow` missing** – install it if you want Parquet output.
* **Files already exist** – add `--overwrite` to replace previous exports.

Happy analysing! 🎧
