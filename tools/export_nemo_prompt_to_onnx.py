#!/usr/bin/env python3
"""Export a NeMo *language-prompt* transducer (EncDecRNNTBPEModelWithPrompt, e.g. datolabs/upelis-0.6b-lt-v1) to the
onnx-asr directory format, with ONE language prompt built into the encoder.

Why a separate script: in these models the language prompt (a one-hot vector concatenated to the encoder output and
projected back by `prompt_kernel`) is applied in the *model's* forward, after the encoder. NeMo exports the subnets
`encoder` and `decoder_joint`, so a plain `model.export()` silently drops the prompt and the ONNX model is no longer
conditioned on the language. Here the encoder gets a subclass whose `forward_for_export` (NeMo swaps the class's
`forward` for it during export) runs the original encoder and then the prompt with a fixed language.

Before exporting, the subclass is checked against `model.forward(..., language_prompt_indices=<lang>)` on a few
real WAV files: any difference above 1e-4 stops the script.

    # needs NeMo 3.1 (unified ASR), e.g.
    # pip install "nemo_toolkit[asr] @ git+https://github.com/NVIDIA-NeMo/Speech.git@8eb458b8ee304143426be5070d3f55fb3b5fb736"
    python tools/export_nemo_prompt_to_onnx.py --nemo upelis-0.6b-lt-v1.nemo --lang lt --out model_fp32 --check a.wav b.wav

Then repack and quantize exactly as for a Parakeet model (see README: the >2 GB encoder is written as per-tensor files).
Measured for LT_ausys (2026-11-15): max |difference| 0.0 on 5 commands; fp32 ONNX = NeMo transcripts on 127/127.
"""
import argparse
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np
import torch
import nemo.collections.asr as nemo_asr


def read_wav(path):
    with wave.open(str(path)) as w:
        assert w.getsampwidth() == 2, f"{path}: 16-bit PCM expected"
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        if w.getnchannels() > 1:
            x = x.reshape(-1, w.getnchannels()).mean(axis=1)
    return x


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--nemo", required=True, help=".nemo checkpoint of a prompt model")
    p.add_argument("--lang", required=True, help="language key of the model's prompt_dictionary, e.g. lt")
    p.add_argument("--out", required=True, help="output directory (onnx-asr format)")
    p.add_argument("--check", nargs="+", required=True, help="16 kHz WAV files for the equality check")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(8)
    m = nemo_asr.models.ASRModel.restore_from(a.nemo, map_location="cpu")
    m.eval()
    if not getattr(m, "concat", False) or not hasattr(m, "prompt_kernel"):
        sys.exit("this model has no language prompt — use export_nemo_to_onnx.py")
    lang = m.resolve_prompt_language(a.lang)
    print(f"{type(m).__name__}; subnets {m.list_export_subnets()}; prompt '{a.lang}' = {lang}")

    base = type(m.encoder)

    class EncoderWithPrompt(base):
        def forward_for_export(self, audio_signal, length, cache_last_channel=None, cache_last_time=None,
                               cache_last_channel_len=None):
            outputs, encoded_lengths = base.forward_for_export(self, audio_signal, length)[:2]
            x = outputs.transpose(1, 2)                                    # B x D x T -> B x T x D
            prompt = self.fixed_prompt.to(x.dtype).expand(x.shape[0], x.shape[1], -1)
            x = self.fixed_prompt_kernel(torch.cat([x, prompt], dim=-1)).to(outputs.dtype)
            return x.transpose(1, 2), encoded_lengths                     # back to B x D x T

    one_hot = torch.zeros(1, 1, m.num_prompts)
    one_hot[0, 0, lang] = 1.0
    m.encoder.register_buffer("fixed_prompt", one_hot, persistent=False)
    m.encoder.add_module("fixed_prompt_kernel", m.prompt_kernel)
    m.encoder.__class__ = EncoderWithPrompt

    worst = 0.0
    with torch.no_grad():
        for wav in a.check:
            sig = torch.from_numpy(read_wav(wav)).unsqueeze(0)
            feats, flen = m.preprocessor(input_signal=sig, length=torch.tensor([sig.shape[1]]))
            ref, _ = m.forward(processed_signal=feats, processed_signal_length=flen,
                               language_prompt_indices=torch.tensor([lang]))
            ours, _ = m.encoder.forward_for_export(feats, flen)
            worst = max(worst, (ref - ours).abs().max().item())
    if worst > 1e-4:
        sys.exit(f"STOP: the prompt encoder differs from model.forward (max |diff| {worst:.2e}) — nothing exported")
    print(f"check OK: max |diff| {worst:.2e} on {len(a.check)} files")

    t0 = time.time()
    m.export(str(out / "model.onnx"), check_trace=False, onnx_opset_version=17)
    print(f"exported in {time.time() - t0:.1f} s")
    with open(out / "vocab.txt", "w", encoding="utf-8") as f:
        for i, tok in enumerate([*m.tokenizer.vocab, "<blk>"]):
            f.write(f"{tok} {i}\n")
    cfg = {"model_type": "nemo-conformer-rnnt", "features_size": int(m.cfg.preprocessor.features),
           "subsampling_factor": int(m.cfg.encoder.subsampling_factor),
           "max_tokens_per_step": int(m.cfg.decoding.greedy.get("max_symbols", 10))}
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print("config.json:", cfg)


if __name__ == "__main__":
    main()
