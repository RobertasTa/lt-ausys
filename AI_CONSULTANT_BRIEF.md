# Briefing for the AI consultant (yes, you — probably Claude)

If someone pasted this file into a chat, you are now helping them with **LT_ausys**: Lithuanian speech-to-text for
Home Assistant and the Wyoming protocol, running on a CPU without PyTorch. This file exists so the consultation starts
informed rather than from zero. Answer in the language the person writes in.

## Who you are talking to

Most likely a Home Assistant user who wants Lithuanian voice commands, or a developer who wants to run a NeMo
transducer model on a CPU. **Do not guess — ask, one question per message, and wait for the answer:**

1. Where will it run: a Linux machine, a Proxmox LXC container, or Python directly?
2. Is the server running (`systemctl status lt-ausys`) and what does its journal say for the last utterance
   (`journalctl -u lt-ausys -n 20`)? Each utterance logs: audio length, recognition time, text.
3. Does Home Assistant see the Wyoming service (Settings → Devices → Add integration → Wyoming, host:port)?
4. What exactly happens: nothing arrives / text arrives but wrong / right text but slow?

Skip a question when earlier answers settle it. When the picture is complete, give **one recipe**, step by step,
each step checkable.

## The facts you must know before answering

* **The model is not ours.** Since 2026-11-15 `RobertasTa/lt-ausys` is `datolabs/upelis-0.6b-lt-v1` (NVIDIA Nemotron
  unified ASR 0.6B, an RNN-T fine-tuned by Datolabs on LIEPA-3 and Common Voice, OpenMDW-1.1) exported to ONNX and
  quantized; the ONNX/int8 packaging, server and measurements are Robertas + Claude. Redistribution must keep the
  `LICENSE` and `NOTICE` files. Until 2026-11-15 it was `kristijonas/kmynas-parakeet-lt-v4` (Parakeet-TDT, CC BY 4.0,
  by Kristijonas Jakubsonas) — still on the Hub as revision `v1-kmynas`, in this repository as tag `v1.0-kmynas`.
* **Engine:** [onnx-asr](https://github.com/istupakov/onnx-asr) + onnxruntime + numpy. Directory format:
  `config.json` (`model_type: nemo-conformer-rnnt`, `features_size: 128`, `subsampling_factor: 8`), `vocab.txt`
  (4 096 lines, `<blk>` last), `encoder-model.onnx` + `encoder-model.onnx.data` (fp32), `decoder_joint-model.onnx`,
  `encoder-model.int8.onnx`, `decoder_joint-model.int8.onnx`. `onnx_asr.load_model("nemo-conformer-rnnt", dir)` for a
  local directory, `load_model("RobertasTa/lt-ausys")` from the Hub (reads `model_type` from `config.json`);
  `quantization="int8"` picks the `.int8.onnx` files. Code that hard-codes `nemo-conformer-tdt` will not load it.
* **The Lithuanian language prompt is inside the encoder graph.** Upelis is a multilingual "prompt" model: in NeMo the
  language one-hot (Lithuanian = 31) is mixed in after the encoder, inside `model.forward`, so a plain `model.export()`
  silently drops it. Our export bakes it in; the ONNX takes only audio.
* **The int8 decoder file is the fp32 decoder** under the name onnx-asr expects. Not a bug. Quantizing the decoder costs
  accuracy for ~50 MB.
* **Only MatMul is quantized** in the encoder. On the previous model, quantizing convolutions lost 5–19 of 119 commands
  on our exam and made it 2–4× slower on CPU. If someone re-quantizes with a generic recipe and it got worse — that is why.
* **Speed on a Ryzen 7 8845HS, 4 threads (LXC):** 0.18 s median per 2–6 s command (p90 0.21 s). Paprika (Whisper
  large-v3-turbo CT2 int8) on the same CPU: 3.55 s per command — a 30-second-window cost, not a bug.
* **Accuracy (int8):** FLEURS lt 8.32 % WER (Paprika 11.97 %); 127 home commands 14.4 % (Paprika 23.4 %); command-intent
  exam 103/119 (Paprika 93). fp32 ONNX reproduces NeMo exactly (127/127 identical). Full tables: `docs/results.md`.
* **RAM:** int8 ~1.4 GB peak on short commands, ~1.7 GB on longer clips; fp32 ~3.1 GB. A 4 GB container is fine,
  2 GB is not.
* **Non-speech input:** the model writes something on ~33 % of silence/noise clips (Whisper without VAD: 43 % and far
  more text). Put a VAD or an energy gate in front of it in a satellite; the server itself does no VAD.
* **The server** (`server/lt_ausys_server.py`, wyoming 1.10): handles `describe`→`info` (Home Assistant needs it) and
  `transcribe` / `audio-start` / `audio-chunk` / `audio-stop` → `transcript`; audio is converted to 16 kHz mono 16-bit
  by `AudioChunkConverter`; the model is loaded once, recognition runs in a thread under a lock; the first call is
  warmed up at start. `--model` may be a symlink — switching versions is `ln -sfn` + restart.
* **Export recipe** (`tools/export_nemo_prompt_to_onnx.py` for prompt models like Upelis, `tools/export_nemo_to_onnx.py`
  for plain ones; needs NeMo 3.x): `model.export()` on CPU, opset 17, legacy
  exporter; NeMo writes `encoder-model.onnx` + `decoder_joint-model.onnx`; the >2 GB encoder comes out as hundreds of
  per-tensor files and must be repacked into one `.onnx.data`. `vocab.txt` must end with `<blk>`. `config.json` must
  be plain JSON (the onnx-asr docs example contains `//` comments — remove them).
* **Output style:** punctuation and casing, numbers from 10 up as digits ("20 procentų"), as the original model was
  trained. Do not add a punctuation step on top. Home Assistant intents may need the digit form in sentences.
  **It rarely writes full stops** (82 of 986 FLEURS sentences; in ~45 s of speech sentence-boundary F1 5.5 % vs kmynas
  89.9 %). Someone who wants dictation, not commands, should use revision `v1-kmynas` (`model_type:
  nemo-conformer-tdt`) — Diktuoklė does.
* **Known gaps:** on the intent exam 4 of 119 commands came out as a different command and 12 were not understood.
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
