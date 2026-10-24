#!/usr/bin/env python3
"""Measure a speech-to-text model the same way for every engine: WER/CER, latency (median, p90), RTF, peak RAM.

Reads a TSV list `path<TAB>reference` (reference empty = non-speech clip: we count invented text instead of WER).
Audio is read with soundfile (any WAV, incl. float32) and passed as float32 numpy at 16 kHz — the engine never sees
the file, so every engine hears exactly the same samples.

    # LT_ausys (onnx-asr), int8, 4 threads
    python bench.py --engine onnx-asr --model model_fp32 --quant int8 --list fleurs.tsv --threads 4 --out results/
    # a CTranslate2 Whisper model (faster-whisper), e.g. Paprika int8
    python bench.py --engine faster-whisper --model paprika-ct2-int8 --list fleurs.tsv --threads 4 --out results/

Greedy decoding, no VAD, Lithuanian forced — the same settings wyoming-faster-whisper uses by default.
Normalisation: lower-case, punctuation removed, Lithuanian letters kept; WER = all errors / all reference words
(not the mean of per-sentence WER). Resumes an interrupted run (rows already in the CSV are skipped).
"""
import argparse
import csv
import json
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf


def normalise(s: str) -> str:
    s = s.lower().replace("„", " ").replace("“", " ").replace("”", " ")
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"_", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def edit_distance(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[len(b)]


def read16k(path):
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        raise SystemExit(f"{path}: {sr} Hz — resample to 16 kHz first")
    return np.ascontiguousarray(audio), len(audio) / sr


class OnnxAsrEngine:
    def __init__(self, model, quant, threads):
        import onnxruntime as rt
        import onnx_asr

        so = rt.SessionOptions()
        so.intra_op_num_threads = threads
        name = model if "/" in model and not Path(model).exists() else "nemo-conformer-tdt"
        path = None if name == model else model
        self.m = onnx_asr.load_model(name, path, quantization=None if quant == "fp32" else quant, sess_options=so)

    def transcribe(self, audio):
        return self.m.recognize(audio, sample_rate=16000).strip()


class FasterWhisperEngine:
    def __init__(self, model, quant, threads):
        from faster_whisper import WhisperModel

        self.m = WhisperModel(model, device="cpu", compute_type="int8" if quant == "int8" else "float32", cpu_threads=threads)

    def transcribe(self, audio):
        segments, _ = self.m.transcribe(audio, language="lt", beam_size=1)
        return " ".join(s.text for s in segments).strip()


def peak_rss_mb():
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    except ImportError:  # Windows
        import psutil

        return psutil.Process().memory_info().peak_wset // 2**20


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", required=True, choices=["onnx-asr", "faster-whisper"])
    p.add_argument("--model", required=True, help="model directory, or a Hugging Face id for onnx-asr")
    p.add_argument("--quant", default="int8", choices=["int8", "fp32"])
    p.add_argument("--list", required=True, help="TSV: path<TAB>reference")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", default="results")
    p.add_argument("--name", help="run name (default: derived)")
    a = p.parse_args()

    rows = []
    with open(a.list, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                path, _, ref = line.rstrip("\n").partition("\t")
                rows.append((Path(path), ref))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name = a.name or f"{Path(a.list).stem}__{Path(a.model).name}__{a.quant}_t{a.threads}"
    csv_path = out / (name + ".csv")
    done = set()
    if csv_path.exists():
        with open(csv_path, encoding="utf-8") as f:
            done = {r["file"] for r in csv.DictReader(f)}
    print(f"=== {name}: {len(rows)} files, {len(done)} already done ===", flush=True)

    t0 = time.perf_counter()
    engine = (OnnxAsrEngine if a.engine == "onnx-asr" else FasterWhisperEngine)(a.model, a.quant, a.threads)
    load_s = time.perf_counter() - t0
    engine.transcribe(read16k(rows[0][0])[0])  # warm-up, not timed
    print(f"  loaded in {load_s:.1f} s, warmed up", flush=True)

    new = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["file", "reference", "hypothesis", "word_errors", "words", "char_errors", "chars", "seconds", "audio_s"])
        for n, (path, ref) in enumerate(rows, 1):
            if str(path) in done:
                continue
            audio, dur = read16k(path)
            t1 = time.perf_counter()
            hyp = engine.transcribe(audio)
            dt = time.perf_counter() - t1
            r, h = normalise(ref), normalise(hyp)
            w.writerow([str(path), ref, hyp, edit_distance(r.split(), h.split()), len(r.split()), edit_distance(r, h), len(r),
                        f"{dt:.4f}", f"{dur:.3f}"])
            f.flush()
            if n % 100 == 0:
                print(f"  [{n}/{len(rows)}]", flush=True)

    with open(csv_path, encoding="utf-8") as f:
        rs = list(csv.DictReader(f))
    times = sorted(float(r["seconds"]) for r in rs)
    audio_total = sum(float(r["audio_s"]) for r in rs)
    s = {"run": name, "engine": a.engine, "model": a.model, "quant": a.quant, "threads": a.threads, "files": len(rs),
         "load_s": round(load_s, 1), "peak_rss_MB": peak_rss_mb(), "median_s": round(statistics.median(times), 3),
         "p90_s": round(times[int(0.9 * (len(times) - 1))], 3), "RTF": round(sum(times) / audio_total, 4) if audio_total else None}
    words = sum(int(r["words"]) for r in rs)
    if words:
        s["WER"] = round(100 * sum(int(r["word_errors"]) for r in rs) / words, 2)
        s["CER"] = round(100 * sum(int(r["char_errors"]) for r in rs) / sum(int(r["chars"]) for r in rs), 2)
    else:  # non-speech set
        with_text = [r for r in rs if normalise(r["hypothesis"])]
        s["clips_with_text_pct"] = round(100 * len(with_text) / len(rs), 1)
        s["invented_chars"] = sum(len(normalise(r["hypothesis"])) for r in rs)
    (out / (name + ".json")).write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SUMMARY:", json.dumps(s, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
