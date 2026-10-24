#!/usr/bin/env python3
"""Talk to a Wyoming speech-to-text server the way Home Assistant does, and time the whole round trip.

    python wyoming_client_test.py --host 192.168.0.232 --port 10302 command1.wav command2.wav ...

Sends `describe` (prints the server's `info`), then for each 16 kHz mono 16-bit WAV: transcribe → audio-start →
audio-chunk… → audio-stop, and waits for `transcript`.
"""
import argparse
import asyncio
import json
import statistics
import time
import wave

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.info import Describe, Info


async def one(host, port, path):
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000 and w.getsampwidth() == 2 and w.getnchannels() == 1, f"{path}: need 16 kHz mono 16-bit"
        pcm = w.readframes(w.getnframes())
    t0 = time.perf_counter()
    async with AsyncTcpClient(host, port) as c:
        await c.write_event(Transcribe(language="lt").event())
        await c.write_event(AudioStart(rate=16000, width=2, channels=1, timestamp=0).event())
        for i in range(0, len(pcm), 8000):
            await c.write_event(AudioChunk(rate=16000, width=2, channels=1, audio=pcm[i:i + 8000], timestamp=i).event())
        await c.write_event(AudioStop(timestamp=len(pcm)).event())
        while True:
            ev = await c.read_event()
            if ev is None:
                return None, time.perf_counter() - t0
            if Transcript.is_type(ev.type):
                return Transcript.from_event(ev).text, time.perf_counter() - t0


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=10302)
    p.add_argument("wavs", nargs="+")
    a = p.parse_args()
    async with AsyncTcpClient(a.host, a.port) as c:
        await c.write_event(Describe().event())
        info = Info.from_event(await c.read_event())
        print("INFO:", json.dumps(info.asr[0].to_dict(), ensure_ascii=False)[:300], "...")
    times = []
    for path in a.wavs:
        text, dt = await one(a.host, a.port, path)
        times.append(dt)
        print(f"{dt:6.3f} s | {path} -> {text!r}")
    if len(times) > 1:
        print(f"round trip: median {statistics.median(times):.3f} s, max {max(times):.3f} s")


if __name__ == "__main__":
    asyncio.run(main())
