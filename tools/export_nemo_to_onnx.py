#!/usr/bin/env python3
"""Export a NeMo Parakeet-TDT / RNNT checkpoint (.nemo) to the onnx-asr directory format.

Needs the NeMo environment (nemo_toolkit[asr] 3.x, torch, onnx) — the ONLY step that does. Runs on CPU:
kmynas v4 (0.6B) exported in 14 s on a desktop. Writes to --out:
    encoder-model.onnx + encoder-model.onnx.data   (NeMo splits the >2 GB encoder into one file per tensor;
                                                     this script repacks them into a single .onnx.data)
    decoder_joint-model.onnx
    vocab.txt      (tokenizer vocabulary + "<blk>" as the last line — required by onnx-asr)
    config.json    (model_type nemo-conformer-tdt, features_size, subsampling_factor, max_tokens_per_step)

    python export_nemo_to_onnx.py --nemo kmynas-v4-last.nemo --out model_fp32
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

KEEP = {"config.json", "vocab.txt", "encoder-model.onnx", "encoder-model.onnx.data", "decoder_joint-model.onnx"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--nemo", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import torch
    import nemo.collections.asr as nemo_asr

    torch.set_num_threads(a.threads)
    t0 = time.perf_counter()
    m = nemo_asr.models.ASRModel.restore_from(a.nemo, map_location="cpu")
    m.eval()
    print(f"loaded in {time.perf_counter() - t0:.1f} s: {type(m).__name__}, subnets {m.list_export_subnets()}")
    is_tdt = getattr(m.cfg.decoding, "model_type", "") == "tdt" or m.joint.num_extra_outputs > 0
    print(f"vocab {m.decoder.vocab_size} | blank {m.decoder.blank_idx} | joint outputs {m.joint.num_classes_with_blank} "
          f"| extra (durations) {m.joint.num_extra_outputs} | tdt={is_tdt}")

    t0 = time.perf_counter()
    m.export(str(out / "model.onnx"), check_trace=False, onnx_opset_version=17)
    print(f"exported in {time.perf_counter() - t0:.1f} s")

    with open(out / "vocab.txt", "w", encoding="utf-8") as f:
        for i, tok in enumerate([*m.tokenizer.vocab, "<blk>"]):
            f.write(f"{tok} {i}\n")
    cfg = {"model_type": "nemo-conformer-tdt" if is_tdt else "nemo-conformer-rnnt",
           "features_size": int(m.cfg.preprocessor.features),
           "subsampling_factor": int(m.cfg.encoder.subsampling_factor), "max_tokens_per_step": 10}
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print("config.json:", cfg)

    # Repack per-tensor external files (torch's legacy exporter does this for models > 2 GB) into one .onnx.data
    import onnx

    enc = out / "encoder-model.onnx"
    model = onnx.load(str(enc))
    ext = [t for t in model.graph.initializer if t.data_location == onnx.TensorProto.EXTERNAL]
    onnx.save(model, str(enc), save_as_external_data=True, all_tensors_to_one_file=True,
              location="encoder-model.onnx.data", size_threshold=1024, convert_attribute=False)
    removed = 0
    for q in out.iterdir():
        if q.name not in KEEP:
            q.unlink()
            removed += 1
    onnx.checker.check_model(str(enc))
    print(f"repacked {len(ext)} external tensors into encoder-model.onnx.data, removed {removed} fragment files")
    for q in sorted(out.iterdir()):
        print(f"  {q.name:28s} {q.stat().st_size:>14,} B")


if __name__ == "__main__":
    sys.exit(main())
