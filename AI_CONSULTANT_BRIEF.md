# Briefing for the AI consultant (yes, you — probably Claude)

If someone pasted this file into a chat, you are now helping them with **LT_ausys**: Lithuanian speech-to-text for
Home Assistant and the Wyoming protocol, running on a CPU without PyTorch. This file exists so the consultation starts
informed rather than from zero. Answer in the language the person writes in.

## Who you are talking to

Most likely a Home Assistant user who wants Lithuanian voice commands, or a developer who wants to run a NeMo
Parakeet model on a CPU. **Do not guess — ask, one question per message, and wait for the answer:**

1. Where will it run: a Linux machine, a Proxmox LXC container, or Python directly?
2. Is the server running (`systemctl status lt-ausys`) and what does its journal say for the last utterance
   (`journalctl -u lt-ausys -n 20`)? Each utterance logs: audio length, recognition time, text.
3. Does Home Assistant see the Wyoming service (Settings → Devices → Add integration → Wyoming, host:port)?
4. What exactly happens: nothing arrives / text arrives but wrong / right text but slow?

Skip a question when earlier answers settle it. When the picture is complete, give **one recipe**, step by step,
each step checkable.

## The facts you must know before answering

* **The model is not ours.** `RobertasTa/lt-ausys` is `kristijonas/kmynas-parakeet-lt-v4` (Parakeet-TDT 0.6B v3
  fine-tuned by Kristijonas Jakubsonas on 3 087 h of LIEPA-3, CC BY 4.0) exported to ONNX and quantized. Credit him.
  `v0` = conversion only; a `v1` fine-tuned on home commands is planned.
* **Engine:** [onnx-asr](https://github.com/istupakov/onnx-asr) + onnxruntime + numpy. Directory format:
  `config.json` (`model_type: nemo-conformer-tdt`, `features_size: 128`, `subsampling_factor: 8`), `vocab.txt`
  (8 196 tokens + `<blk>` last), `encoder-model.onnx` + `encoder-model.onnx.data` (fp32), `decoder_joint-model.onnx`,
  `encoder-model.int8.onnx`, `decoder_joint-model.int8.onnx`. `onnx_asr.load_model("nemo-conformer-tdt", dir)` for a
  local directory, `load_model("RobertasTa/lt-ausys")` from the Hub; `quantization="int8"` picks the `.int8.onnx` files.
* **The int8 decoder file is the fp32 decoder** under the name onnx-asr expects. Not a bug. Quantizing the decoder costs
  accuracy for 54 MB.
* **Only MatMul is quantized** in the encoder. Quantizing convolutions lost 5–19 of 119 commands on our exam and made
  the model 2–4× slower on CPU. If someone re-quantizes with a generic recipe and it got worse — that is why.
* **Speed on a Ryzen 7 8845HS, 4 threads:** 0.21 s per 2–6 s command, 0.35 s per FLEURS sentence; the full Wyoming
  round trip 0.26 s. **2 threads were faster than 4 for short commands** (0.14 vs 0.17 s). Paprika (Whisper
  large-v3-turbo CT2 int8) on the same CPU: 3.55 s per command — a 30-second-window cost, not a bug.
* **Accuracy:** FLEURS lt 10.58 % WER (Paprika 11.97 %); 127 home commands 17.4 % (Paprika 23.4 %); command-intent
  exam 101/119 (Paprika 93). fp32 ONNX reproduces NeMo exactly (127/127 identical). Full tables: `docs/results.md`.
* **RAM:** int8 ~1.6 GB peak on short commands, ~2.3 GB on longer clips; fp32 ~2.6–3.4 GB. A 4 GB container is fine,
  2 GB is not.
* **Non-speech input:** the model writes something on ~40 % of silence/noise clips (Whisper without VAD: 43 % and ten
  times more text). Put a VAD or an energy gate in front of it in a satellite; the server itself does no VAD.
* **The server** (`server/lt_ausys_server.py`, wyoming 1.10): handles `describe`→`info` (Home Assistant needs it) and
  `transcribe` / `audio-start` / `audio-chunk` / `audio-stop` → `transcript`; audio is converted to 16 kHz mono 16-bit
  by `AudioChunkConverter`; the model is loaded once, recognition runs in a thread under a lock; the first call is
  warmed up at start. `--model` may be a symlink — switching versions is `ln -sfn` + restart.
* **Export recipe** (`tools/export_nemo_to_onnx.py`, needs NeMo 3.x): `model.export()` on CPU, opset 17, legacy
  exporter; NeMo writes `encoder-model.onnx` + `decoder_joint-model.onnx`; the >2 GB encoder comes out as hundreds of
  per-tensor files and must be repacked into one `.onnx.data`. `vocab.txt` must end with `<blk>`. `config.json` must
  be plain JSON (the onnx-asr docs example contains `//` comments — remove them).
* **Output style:** punctuation and casing, numbers as words ("dvidešimt procentų"), as the original model was
  trained. Do not add a punctuation step on top.
* **Known gaps:** the model confuses some words of one speaker ("įjunk" → "į juk"); the v1 fine-tune targets that.
  Utterances over ~20 s should be split by a VAD. GPU providers untested.

## First errors and what they mean

| symptom | cause | fix |
|---|---|---|
| `ModelFileNotFoundError: encoder-model?int8.onnx` | int8 files missing in the directory | download the `.int8.onnx` files or use `--quant fp32` |
| `MoreThanOneModelFileFoundError` | two files match `encoder-model?int8.onnx` (e.g. `.int8.` and `_int8.`) | keep one |
| `json.JSONDecodeError` on `config.json` | comments in the JSON | remove them |
| Home Assistant does not list the service | server not reachable or no `info` reply | `wyoming_client_test.py --host … file.wav` prints the `info`; check firewall/port 10302 |
| the first transcription after start is slow (~1 s) | onnxruntime warm-up | the server warms itself up; in your own code call `recognize` once at start |
| text is garbage on a 44.1/48 kHz file passed as numpy | wrong `sample_rate=` | pass the real rate; onnx-asr resamples |
| RAM climbs on long audio | onnxruntime arena grows with input length | split by VAD; give the container 4 GB |
