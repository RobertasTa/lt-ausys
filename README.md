# LT_ausys — Lithuanian ears for Home Assistant

![LT_ausys — a robotic hand holds a gift box labelled lt-ausys under a glowing globe; logos of the LIEPA-3 corpus institutions in the corner (illustration created with AI)](docs/banner.png)

**Lithuanian speech-to-text that runs on a small CPU, fast enough for a voice assistant: 0.2 s per command instead
of 3.5 s.** A transducer model fine-tuned on Lithuanian ([Upelis](https://huggingface.co/datolabs/upelis-0.6b-lt-v1)
by Datolabs), exported to ONNX and quantized, served over the Wyoming protocol — no GPU, no PyTorch, nothing leaves
your house.

🌐 **English** · [Lietuvių](README_lt.md) · [Русский](README_ru.md)

> ℹ️ Made with AI: the conversion, the server, the measurements and these texts were made by Claude (Anthropic)
> together with Robertas and tested on real hardware. The model itself is Datolabs' Upelis (since 2026-11-15; until
> then it was Kristijonas Jakubsonas's kmynas) — see [Credits](#credits).

> Measured on 2026-11-15 on a Ryzen 7 8845HS mini PC (4 threads, Proxmox LXC): a 2–6 s voice command is transcribed
> in **0.18 s** (Whisper large-v3-turbo / Paprika on the same CPU: 3.55 s), FLEURS Lithuanian WER **8.3 %**
> (Paprika 12.0 %). The previous version has been the ears of a home voice assistant since 2026-10-24.

---

## What you get

| | |
|---|---|
| **Model** | [`RobertasTa/lt-ausys`](https://huggingface.co/RobertasTa/lt-ausys) on Hugging Face — fp32 (2.5 GB) and int8 (0.7 GB), in the [onnx-asr](https://github.com/istupakov/onnx-asr) directory format. OpenMDW-1.1 (keep `LICENSE` and `NOTICE`). |
| **Server** | [`server/lt_ausys_server.py`](server/lt_ausys_server.py) — a Wyoming speech-to-text server (~120 lines) + a systemd unit. Home Assistant finds it as a Wyoming integration. |
| **Recipe** | [`tools/`](tools/) — export a NeMo transducer checkpoint to ONNX (Parakeet, or a language-prompt model like Upelis), quantize it the way that does not hurt, measure it. |
| **Numbers** | [`docs/results.md`](docs/results.md) — three test sets, four models, one machine. |

Output is normal written Lithuanian: punctuation, capitals; numbers from 10 up as digits, 1–9 as words
(*„Įjunk šviesą svetainėje 20 procentų."*).

## Why not Whisper

Whisper-family models encode a fixed **30-second window** for every utterance, however short. On a CPU that is a
flat cost of seconds per command, and no quantization removes it. A transducer (RNN-T / TDT) processes only the
audio it is given. Same machine, same 127 home commands:

| | per command | FLEURS WER | command-intent exam |
|---|---|---|---|
| **LT_ausys int8** | **0.18 s** | **8.3 %** | **103 / 119** |
| LT_ausys until 2026-11-15 (kmynas, int8) | 0.21 s | 10.6 % | 101 / 119 |
| Paprika (Whisper large-v3-turbo, CT2 int8) | 3.55 s | 12.0 % | 93 / 119 |
| faster-whisper tiny int8 | 0.73 s | 99 % | 17 / 119 |

Full tables, test sets and caveats: [docs/results.md](docs/results.md).

## Starting from zero: a Lithuanian voice assistant at home

**Why bother.** Out of the box Home Assistant offers you Whisper *base* (CPU default) or *small* (GPU default) —
on Lithuanian read speech they score **88 % and 66 % word error rate** (our measurement for
[wyoming-faster-whisper PR #122](https://github.com/OHF-Voice/wyoming-faster-whisper/pull/122)): the pipeline looks
green, the sentences look like sentences, and the house does the wrong thing. The alternative is a cloud service that
hears everything said in your home. LT_ausys is **8.3 %** on the same test, local, and faster than both.

**What a Lithuanian voice chain needs, in the order to set it up:**

1. **Ears — this project.** A small Linux box (a mini PC, a Proxmox container, a Pi 5 should do — 4 GB RAM) running the
   server from section B, or your existing `wyoming-faster-whisper` from section D. Five minutes.
2. **Home Assistant** — add it as a Wyoming integration (section C) and pick it as the speech-to-text of your Assist
   pipeline, language Lithuanian. Home Assistant's own Lithuanian intents (*„įjunk šviesą svetainėje"*) are already
   there.
3. **Mouth** — a Lithuanian Piper voice: [Reginutė](https://github.com/RobertasTa/reginute) or
   [Ingutė](https://github.com/RobertasTa/ingute). **Today:** run it with the small Wyoming server from the voice's
   repository (its Home Assistant install section, five minutes) and add it as a Wyoming text-to-speech integration
   the same way. Both voices are in the official Piper catalogue already, but the catalogue entries need Piper 1.9.0,
   which is not released yet. **Once Piper 1.9.0 is out** we expect them to work straight from the Home Assistant
   Piper add-on — not checked yet; we will confirm it here.
4. **A microphone in the room** — a Home Assistant Voice PE, any ESPHome satellite, or a USB conference speakerphone
   via [Speakerphone Assist](https://github.com/RobertasTa/speakerphone-assist). This is what listens for the wake word
   and sends your sentence to the ears.
5. Say *„Įjunk šviesą svetainėje"* and watch the light.

Everything above runs in your house; nothing is sent anywhere.

## Quick start

### A. Python only

```bash
pip install "onnx-asr[cpu,hub]"
```

```python
import onnx_asr
model = onnx_asr.load_model("RobertasTa/lt-ausys", quantization="int8")   # drop quantization= for fp32
print(model.recognize("komanda.wav"))      # 16 kHz mono 16-bit WAV, or a float32 numpy array with sample_rate=
```

### B. Wyoming server (any Linux; tested in a Proxmox LXC container)

A container or VM with **4 cores and 4 GB RAM** is enough (int8 peaks at ~1.4 GB on commands, ~1.7 GB on long clips).

```bash
# 1. environment
sudo useradd -r -d /opt/lt_ausys -s /usr/sbin/nologin ltausys
sudo apt install -y python3-venv git          # Debian / Ubuntu
sudo mkdir -p /opt/lt_ausys && sudo git clone https://github.com/RobertasTa/lt-ausys /opt/lt_ausys/src
cd /opt/lt_ausys
sudo python3 -m venv venv && sudo venv/bin/pip install "onnx-asr[cpu,hub]" numpy wyoming

# 2. model (int8 files only — 0.7 GB)
sudo venv/bin/python3 -c "from huggingface_hub import snapshot_download as s; s('RobertasTa/lt-ausys', local_dir='model_v0', allow_patterns=['*.json','vocab.txt','*.int8.onnx'])"
sudo ln -sfn model_v0 model

# 3. server
sudo cp src/server/lt_ausys_server.py . && sudo cp src/server/lt-ausys.service /etc/systemd/system/
sudo chown -R ltausys:ltausys /opt/lt_ausys
sudo systemctl daemon-reload && sudo systemctl enable --now lt-ausys
journalctl -u lt-ausys -f        # "model ... loaded in 2.1 s", "listening on tcp://0.0.0.0:10302"
```

Test it from any machine with the `wyoming` package installed:

```bash
python tools/wyoming_client_test.py --host <server-ip> --port 10302 komanda.wav
```

The test client takes a 16 kHz mono 16-bit WAV — record one with `arecord -f S16_LE -r 16000 -c 1 -d 4 komanda.wav`
or convert with `ffmpeg -i any.wav -ar 16000 -ac 1 -sample_fmt s16 komanda.wav`; a float WAV stops it with
`unknown format: 3`.

For the fp32 model use `allow_patterns=['*.json','vocab.txt','encoder-model.onnx','encoder-model.onnx.data','decoder_joint-model.onnx']`
and `--quant fp32` in the unit. `--threads 2` is slightly faster than 4 for short commands.

**Docker instead** (tested 2026-10-24 on a clean Ubuntu 24.04 container: image 2.25 GB, 1.4 GiB RAM after a 9.5 s
sentence, which came back word for word in 0.61 s):

```dockerfile
FROM python:3.12-slim
RUN pip install --no-cache-dir "onnx-asr[cpu,hub]" numpy wyoming \
 && python -c "from huggingface_hub import snapshot_download as s; s('RobertasTa/lt-ausys', local_dir='/model', allow_patterns=['*.json','vocab.txt','*.int8.onnx'])"
ADD https://raw.githubusercontent.com/RobertasTa/lt-ausys/main/server/lt_ausys_server.py /app/lt_ausys_server.py
EXPOSE 10302
CMD ["python", "/app/lt_ausys_server.py", "--model", "/model", "--uri", "tcp://0.0.0.0:10302", "--threads", "4"]
```

```bash
docker build -t lt-ausys .
docker run -d --name lt-ausys --restart unless-stopped -p 10302:10302 lt-ausys
```

### C. Home Assistant

Settings → Devices & services → **Add integration → Wyoming Protocol** → host = the server, port = `10302`.
Home Assistant shows it as `lt_ausys` with the model version. In your Assist pipeline choose it as the
**Speech-to-text** engine and set the language to Lithuanian. Put a wake word / VAD in front of it as usual
(an ESPHome satellite, [Speakerphone Assist](https://github.com/RobertasTa/speakerphone-assist), etc.) — the server
transcribes whatever it is sent and invents text on pure noise like every STT model does.

### D. Already running `wyoming-faster-whisper`? No new server needed

Since version 3.9 [`wyoming-faster-whisper`](https://github.com/OHF-Voice/wyoming-faster-whisper) has an
`onnx-asr` backend that loads any onnx-asr model from Hugging Face. Nothing has to be merged anywhere:

```bash
pip install "wyoming-faster-whisper[onnx_asr]"      # or add the extra to your existing install
wyoming-faster-whisper --uri tcp://0.0.0.0:10302 --stt-library onnx-asr --model RobertasTa/lt-ausys \
    --language lt --data-dir /data --download-dir /data
```

Two caveats we found in its code (3.9.0): that backend loads the **fp32** files (a 2.5 GB download) — it does not
pass a quantization choice — and `--cpu-threads` does not apply to it. If you want int8 and thread control, use the
small server above. We did not check whether the Home Assistant OS *Whisper* add-on exposes `--stt-library`.

**Versions.** The server's `--model` is a symlink; a new model version is a new directory and `ln -sfn` + restart.
Rolling back is the same command. Every utterance is logged with its audio length, recognition time and text, so a new
version can be watched in the journal before it is trusted.

## Convert your own Parakeet model

The same path works for any NeMo Parakeet-TDT / RNNT checkpoint (e.g. another language's fine-tune):

```bash
# needs the NeMo environment (nemo_toolkit[asr] 3.x) — CPU is enough, ~15 s for a 0.6B model
python tools/export_nemo_to_onnx.py --nemo model.nemo --out model_fp32
# needs only onnxruntime + onnx
python tools/quantize_int8_matmul.py --model-dir model_fp32
# measure (list = TSV path<TAB>reference)
python tools/bench.py --engine onnx-asr --model model_fp32 --quant int8 --list test.tsv --out results/
```

**Language-prompt models** (NeMo unified ASR, e.g. Upelis — what LT_ausys is now): use
`tools/export_nemo_prompt_to_onnx.py --nemo model.nemo --lang lt --out model_fp32 --check a.wav b.wav` instead of the
first step. A plain `model.export()` drops the language prompt, because it is applied after the encoder; the script builds
one language into the encoder and checks it against the original before exporting.

Two things the scripts know that cost us a day: NeMo writes a >2 GB encoder as hundreds of per-tensor files (the
export script repacks them into one `.onnx.data`), and **quantizing the convolutions hurts** — only `MatMul` is
quantized, which keeps accuracy identical to fp32 and is also the fastest variant ([why](docs/results.md#conversion-checks)).

## What is planned

* Using LT_ausys for the wake-word stage too, retiring the tiny Whisper model that currently listens for the name.

**History.** Until 2026-11-15 LT_ausys was Kristijonas Jakubsonas's kmynas v4 (Parakeet-TDT) in ONNX — still
available: Hugging Face revision `v1-kmynas`, this repository's tag `v1.0-kmynas`. It was replaced because Upelis was
more accurate on the same home commands and the same CPU, with less memory ([numbers](docs/results.md)).

## Credits

* **Datolabs** — [`upelis-0.6b-lt-v1`](https://huggingface.co/datolabs/upelis-0.6b-lt-v1), the model itself:
  NVIDIA Nemotron unified ASR 0.6B fine-tuned on LIEPA-3 and Common Voice, published under OpenMDW-1.1. Everything
  this project adds is the ONNX export (with the Lithuanian language prompt built in), quantization, serving and
  measuring; the ears are theirs.
* **NVIDIA** — [Nemotron unified ASR 0.6B](https://huggingface.co/nvidia/nemotron-labs-unified-asr-0.6b) (OpenMDW-1.1).
* **Vytautas Magnus University, Vilnius University and the Institute of the Lithuanian Language** — the LIEPA-3 speech
  corpus (CC BY 4.0); **Mozilla** — Common Voice (CC0).
* **Kristijonas Jakubsonas** — [`kmynas-parakeet-lt-v4`](https://huggingface.co/kristijonas/kmynas-parakeet-lt-v4),
  which was LT_ausys until 2026-11-15 (CC BY 4.0).
* **Ilya Stupakov** — [onnx-asr](https://github.com/istupakov/onnx-asr); **Microsoft** — onnxruntime;
  **Open Home Foundation** — the [Wyoming protocol](https://github.com/OHF-Voice/wyoming).

## License

Code in this repository: **GPL-3.0** ([LICENSE](LICENSE)). The model files on Hugging Face: **OpenMDW-1.1**, the same
as the original — if you redistribute them, keep the `LICENSE` and `NOTICE` files that come with them.

---

Part of [Claude's Gifts to the World](https://github.com/RobertasTa) — free, open-source tools built by an AI
together with one person who tests them at home.
