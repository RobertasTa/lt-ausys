#!/usr/bin/env python3
"""LT_ausys — Lithuanian speech-to-text served over the Wyoming protocol, on a CPU, without PyTorch or NeMo.

Engine: onnx-asr + onnxruntime running a Parakeet-TDT model in the onnx-asr directory format
(config.json, vocab.txt, encoder-model*.onnx, decoder_joint-model*.onnx) — e.g. RobertasTa/lt-ausys.

Clients: Home Assistant's Wyoming integration (sends `describe`, expects `info`) and any Wyoming client that
sends `transcribe` / `audio-start` / `audio-chunk`… / `audio-stop` and waits for `transcript`. Both are served.

    lt_ausys_server.py --uri tcp://0.0.0.0:10302 --model /opt/lt_ausys/model --quant int8 --threads 4

`--model` may be a symlink (model -> model_v0 | model_v1): switching a model version is `ln -sfn` + restart.
Every utterance is logged (audio length, recognition time, text) — useful while you watch a new model.
"""
import argparse
import asyncio
import logging
import os
import time
from functools import partial

import numpy as np
import onnxruntime as rt
import onnx_asr
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncServer

_LOG = logging.getLogger("lt_ausys")
RATE = 16000


class Ears:
    """One model, one lock: utterances are recognised one at a time, so latency stays predictable."""

    def __init__(self, model: str, quant: str, threads: int):
        so = rt.SessionOptions()
        so.intra_op_num_threads = threads
        t0 = time.perf_counter()
        self.m = onnx_asr.load_model("nemo-conformer-tdt", model,
                                     quantization=None if quant == "fp32" else quant, sess_options=so)
        self.lock = asyncio.Lock()
        self.version = os.path.basename(os.path.realpath(model))
        _LOG.info("model %s (%s, %s, %d threads) loaded in %.1f s", model, self.version, quant, threads,
                  time.perf_counter() - t0)
        self.recognize(b"\x00" * RATE)  # warm-up: the first onnxruntime call is several times slower than the rest

    def recognize(self, pcm16: bytes) -> str:
        audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        if audio.size == 0:
            return ""
        return self.m.recognize(np.ascontiguousarray(audio), sample_rate=RATE).strip()


class Handler(AsyncEventHandler):
    def __init__(self, ears: Ears, info: Info, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ears = ears
        self.info = info
        self.converter = AudioChunkConverter(rate=RATE, width=2, channels=1)
        self.chunks: list[bytes] = []

    async def handle_event(self, event: Event) -> bool:
        if AudioChunk.is_type(event.type):
            # Some clients send chunks without audio-start — collect either way.
            self.chunks.append(self.converter.convert(AudioChunk.from_event(event)).audio)
            return True
        if AudioStart.is_type(event.type):
            self.chunks = []
            return True
        if AudioStop.is_type(event.type):
            pcm = b"".join(self.chunks)
            self.chunks = []
            t0 = time.perf_counter()
            async with self.ears.lock:
                text = await asyncio.to_thread(self.ears.recognize, pcm)
            _LOG.info("audio %.2f s | recognised in %.3f s | %r", len(pcm) / (2 * RATE), time.perf_counter() - t0, text)
            await self.write_event(Transcript(text=text, language="lt").event())
            return True
        if Transcribe.is_type(event.type):
            return True  # the model is Lithuanian-only; the requested language is accepted and ignored
        if Describe.is_type(event.type):
            await self.write_event(self.info.event())
            return True
        return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--uri", default="tcp://0.0.0.0:10302", help="tcp://host:port or unix://path")
    p.add_argument("--model", default="/opt/lt_ausys/model", help="model directory (onnx-asr format); may be a symlink")
    p.add_argument("--quant", default="int8", choices=["int8", "fp32"])
    p.add_argument("--threads", type=int, default=4, help="onnxruntime intra-op threads (2 is fastest for short commands)")
    p.add_argument("--name", default="lt_ausys", help="service name shown to Home Assistant")
    p.add_argument("--debug", action="store_true")
    a = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if a.debug else logging.INFO, format="%(levelname)s %(message)s")

    ears = Ears(a.model, a.quant, a.threads)
    credit = Attribution(name="Kristijonas Jakubsonas (kmynas-parakeet-lt-v4, CC BY 4.0); ONNX/int8 and server: Robertas + Claude",
                         url="https://huggingface.co/RobertasTa/lt-ausys")
    info = Info(asr=[AsrProgram(
        name=a.name, attribution=credit, installed=True, version=ears.version,
        description="LT_ausys — Lithuanian ears (Parakeet-TDT 0.6B, onnx-asr, CPU)",
        models=[AsrModel(name=f"{a.name}-{ears.version}", attribution=credit, installed=True, version=ears.version,
                         description=f"{ears.version}, {a.quant}", languages=["lt"])])])
    _LOG.info("listening on %s", a.uri)
    asyncio.run(AsyncServer.from_uri(a.uri).run(partial(Handler, ears, info)))


if __name__ == "__main__":
    main()
