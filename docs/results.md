# Measurements — LT_ausys v0 against Paprika and faster-whisper tiny

All numbers on **one machine**: AMD Ryzen 7 8845HS (a mini PC), inside an unprivileged Proxmox LXC container with
4 cores, 4 threads per engine, greedy decoding (beam 1), no VAD, Lithuanian forced. Measured 2026-10-24 with
[`tools/bench.py`](../tools/bench.py). Every engine got exactly the same float32 samples; the file was never passed
to the engine.

* **LT_ausys v0** — `RobertasTa/lt-ausys` (kmynas v4 exported to ONNX), int8 = MatMul-only quantization.
* **Paprika** — `kristijonas/paprika-whisper-lt-v3` (Whisper large-v3-turbo, 3 281 h LIEPA-3), CTranslate2 int8, the
  same files `wyoming-faster-whisper` runs.
* **tiny** — `faster-whisper-tiny` int8, the generic model a Home Assistant whisper add-on starts with on a CPU.

## Three test sets

| set | what | why |
|---|---|---|
| **FLEURS lt test** — 986 read sentences, 17 136 reference words | public, neutral (neither model was trained on it) | the comparable accuracy figure |
| **127 home commands** — one speaker, a conference speakerphone (EMEET Luna Plus), a living room | what a voice assistant actually hears | latency and accuracy on short commands |
| **500 non-speech clips** (`akisviete/azuolas-nonspeech-pool`, 125 from each of 4 sources) | silence, noise, music | how much text the model invents when nobody speaks |

Normalisation: lower-case, punctuation removed, Lithuanian letters kept. **WER = sum of errors / sum of reference
words** over the whole set (not the mean of per-sentence WER). Latency = median and 90th percentile per utterance,
engine already loaded and warmed up. RTF = total processing time / total audio.

## Results

| model | FLEURS: WER · CER · median · p90 · RTF | commands: WER · median · p90 | non-speech: clips with text · characters invented · median time | peak RAM |
|---|---|---|---|---|
| **LT_ausys v0 int8** | **10.58 %** · 6.15 % · **0.35 s** · 0.55 s · 0.035 | **17.4 %** · **0.21 s** · 0.26 s | 40 % · 5 608 · 0.28 s | 2.3 GB |
| LT_ausys v0 fp32 | 10.55 % · 6.13 % · 0.59 s · 0.97 s · 0.059 | 18.1 % · 0.24 s · 0.37 s | 34 % · 4 471 · 0.49 s | 3.4 GB |
| Paprika (CT2 int8) | 11.97 % · 6.49 % · 3.90 s · 4.12 s · 0.363 | 23.4 % · 3.55 s · 3.61 s | 43 % · **19 099** · **7.2 s** | 1.4 GB |
| faster-whisper tiny int8 | 99.4 % · 38.1 % · 0.79 s · 2.39 s · 0.107 | 102.6 % · 0.73 s · 0.94 s | 13 % · 1 756 · 3.1 s | 0.3 GB |

### What the table says

* **Commands: 17× faster than Paprika and more accurate.** 0.21 s against 3.55 s for the same 2–6 s utterances.
  Whisper encodes a fixed 30-second window whatever the input length; a transducer processes only the audio it gets.
* **Read speech: more accurate too** (10.58 % vs 11.97 %), 11× faster.
* **Non-speech: Paprika talks.** Without VAD, Whisper writes long invented sentences on noise and music (72 of the
  500 clips produced more than 80 characters), and every invented token costs decoding time — hence a median of
  7.2 s and a maximum of 22 s per clip. LT_ausys invents 3.4× less text and stays at 0.3 s.
* **tiny is not a Lithuanian model.** 99 % WER on read speech: it recognises a wake word's shape, not sentences.

### Command exam (the assistant's point of view)

The 119 commands with a defined intent were also fed through the home assistant's command dictionary, so the question
becomes "would the assistant have done the right thing?" rather than "how many words differ":

| ears (transcripts from the table above) | correct intent | wrong intent | not understood |
|---|---|---|---|
| LT_ausys v0 int8 | **101** | 3 | 15 |
| LT_ausys v0 fp32 | 101 | 3 | 15 |
| Paprika | 93 | 5 | 21 |
| tiny | 17 | 1 | 101 |

The dictionary was tuned for Paprika's typical mistakes over several weeks; LT_ausys scores higher before any tuning.

## Conversion checks

* **fp32 ONNX = the NeMo checkpoint.** 127/127 command transcripts identical, character for character, to
  `ASRModel.transcribe()` of the original `.nemo` (NeMo 3.0.0, CPU).
* **int8 is deterministic across CPUs.** The int8 transcripts are bit-identical on a Zen 5 desktop (Ryzen 9 9950X)
  and the Zen 4 mini PC above. faster-whisper tiny int8 is not: its texts changed between the same two machines.
* **Which int8 recipe** (desktop, 127 commands, exam out of 119):

| encoder quantization (decoder fp32 unless noted) | exam | ≠ fp32 decisions | WER | median (desktop) |
|---|---|---|---|---|
| **QInt8 per channel, MatMul only** | **101** | 0 | 17.4 % | **0.045 s** |
| QUInt8, MatMul only | 100 | 1 | 18.7 % | 0.053 s |
| QUInt8 all ops + decoder QInt8 (the recipe used for sherpa-onnx Parakeet exports) | 96 | 8 | 22.4 % | 0.052 s |
| QInt8 per channel, all ops | 91 | 14 | 23.7 % | 0.183 s |
| QInt8 per channel, all ops, after `quant_pre_process` | 91 | 14 | 23.7 % | 0.182 s |
| QInt8 per channel, all ops + decoder QInt8 | 91 | 13 | 23.4 % | 0.173 s |
| QInt8 per channel, all ops, `reduce_range` | 82 | 21 | 28.8 % | 0.182 s |

  Quantizing the convolutions is what hurts — and it also makes the model 2–4× slower on this CPU.
  Quantizing the decoder on top of the MatMul-only encoder: 100/119 for 54 MB saved — not taken.

## Caveats

* The command set is **one speaker**. It shows the behaviour on short home commands, not generalisation across voices.
* **18 % of FLEURS references write numbers as digits** the speaker never said. Every model here writes numbers as
  words, so all of them are penalised equally; absolute figures are pessimistic for all.
* Whisper with VAD (`vad_filter`) invents far less on non-speech; `wyoming-faster-whisper` runs without it by default,
  which is what we measured.
* Not measured: GPU, batching, audio longer than 20 s, other speakers.
