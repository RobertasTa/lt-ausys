#!/usr/bin/env python3
"""int8 dynamic quantization of the encoder — MatMul only (QInt8, per channel). Needs onnxruntime + onnx, no NeMo.

Why only MatMul: on kmynas v4 the usual "quantize everything" recipe (encoder QUInt8, decoder QInt8) lost 5 of 119
home commands on our exam and every variant that quantized the convolutions lost 5–19 commands AND ran 2–4× slower
on the CPU. Quantizing the matrix multiplications only keeps the exam identical to fp32 (101/119) and is the fastest
variant. The decoder is left in fp32: quantizing it costs one command for 54 MB.

Writes next to the fp32 files:
    encoder-model.int8.onnx          (882 MB for kmynas v4, from 2.48 GB)
    decoder_joint-model.int8.onnx    (a copy of the fp32 decoder under the name onnx-asr expects for quantization="int8";
                                      pass --decoder-int8 to really quantize it)

    python quantize_int8_matmul.py --model-dir model_fp32
"""
import argparse
import shutil
import time
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--decoder-int8", action="store_true")
    a = p.parse_args()
    d = Path(a.model_dir)
    t0 = time.perf_counter()
    quantize_dynamic(d / "encoder-model.onnx", d / "encoder-model.int8.onnx",
                     weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=["MatMul"])
    print(f"encoder: {(d / 'encoder-model.int8.onnx').stat().st_size:,} B in {time.perf_counter() - t0:.1f} s")
    if a.decoder_int8:
        quantize_dynamic(d / "decoder_joint-model.onnx", d / "decoder_joint-model.int8.onnx",
                         weight_type=QuantType.QInt8, per_channel=True)
        print("decoder: quantized (QInt8 per channel)")
    else:
        shutil.copy2(d / "decoder_joint-model.onnx", d / "decoder_joint-model.int8.onnx")
        print("decoder: fp32 copy under the int8 name")


if __name__ == "__main__":
    main()
