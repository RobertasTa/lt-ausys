# LT_ausys — литовские уши для Home Assistant

**Распознавание литовской речи на небольшом процессоре, достаточно быстрое для голосового ассистента: 0,2 с на
команду вместо 3,5 с.** Модель Parakeet-TDT, дообученная на 3 087 часах литовской речи, экспортированная в ONNX и
квантованная, отдаётся по протоколу Wyoming — без видеокарты, без PyTorch, ничего не покидает ваш дом.

🌐 [English](README.md) · [Lietuvių](README_lt.md) · **Русский**

> ℹ️ Сделано с ИИ: конвертацию, сервер, измерения и эти тексты делал Claude (Anthropic) вместе с Робертасом, проверено
> на реальном железе. Сама модель — Кристийонаса Якубсонаса, см. [Благодарности](#благодарности).

> Измерено 2026-10-24 на мини-ПК с Ryzen 7 8845HS (4 потока, Proxmox LXC): голосовая команда длиной 2–6 с
> распознаётся за **0,21 с** (Whisper large-v3-turbo / Paprika на том же процессоре: 3,55 с), WER на FLEURS lt —
> **10,6 %** (Paprika 12,0 %). С того вечера — уши домашнего голосового ассистента.

---

## Что вы получаете

| | |
|---|---|
| **Модель** | [`RobertasTa/lt-ausys`](https://huggingface.co/RobertasTa/lt-ausys) на Hugging Face — fp32 (2,5 ГБ) и int8 (0,9 ГБ) в формате каталога [onnx-asr](https://github.com/istupakov/onnx-asr). CC BY 4.0. |
| **Сервер** | [`server/lt_ausys_server.py`](server/lt_ausys_server.py) — Wyoming-сервер распознавания речи (~120 строк) + unit systemd. Home Assistant видит его как интеграцию Wyoming. |
| **Рецепт** | [`tools/`](tools/) — экспорт любой NeMo Parakeet-модели в ONNX, квантование «которое не портит», измерение. |
| **Цифры** | [`docs/results.md`](docs/results.md) — три тестовых набора, четыре модели, одна машина. |

На выходе — обычный письменный литовский: пунктуация, заглавные, числа словами (*„Įjunk šviesą svetainėje
dvidešimt procentų."*).

## Почему не Whisper

Модели семейства Whisper на каждую фразу, какой бы короткой она ни была, кодируют фиксированное **30-секундное
окно**. На процессоре это постоянная плата в несколько секунд за команду, и никакое квантование её не снимает.
Трансдьюсер (Parakeet-TDT) обрабатывает только тот звук, который ему дали. Та же машина, те же 127 домашних команд:

| | на команду | WER FLEURS | экзамен по намерению команды |
|---|---|---|---|
| **LT_ausys v0 int8** | **0,21 с** | **10,6 %** | **101 / 119** |
| Paprika (Whisper large-v3-turbo, CT2 int8) | 3,55 с | 12,0 % | 93 / 119 |
| faster-whisper tiny int8 | 0,73 с | 99 % | 17 / 119 |

Полные таблицы, наборы и оговорки: [docs/results.md](docs/results.md).

## Быстрый старт

### A. Только Python

```bash
pip install "onnx-asr[cpu,hub]"
```

```python
import onnx_asr
model = onnx_asr.load_model("RobertasTa/lt-ausys", quantization="int8")   # без quantization= — fp32
print(model.recognize("komanda.wav"))      # WAV 16 кГц моно 16 бит или массив float32 numpy с sample_rate=
```

### B. Сервер Wyoming (любой Linux; проверено в контейнере Proxmox LXC)

Достаточно контейнера или VM с **4 ядрами и 4 ГБ RAM** (пик int8 ~1,6 ГБ на командах, ~2,3 ГБ на длинных записях).

```bash
# 1. окружение
sudo useradd -r -d /opt/lt_ausys -s /usr/sbin/nologin ltausys
sudo mkdir -p /opt/lt_ausys && cd /opt/lt_ausys
sudo python3 -m venv venv && sudo venv/bin/pip install "onnx-asr[cpu,hub]" numpy wyoming

# 2. модель (только int8-файлы — 0,9 ГБ)
sudo venv/bin/python3 -c "from huggingface_hub import snapshot_download as s; s('RobertasTa/lt-ausys', local_dir='model_v0', allow_patterns=['*.json','vocab.txt','*.int8.onnx'])"
sudo ln -sfn model_v0 model

# 3. сервер
sudo cp server/lt_ausys_server.py . && sudo cp server/lt-ausys.service /etc/systemd/system/
sudo chown -R ltausys:ltausys /opt/lt_ausys
sudo systemctl daemon-reload && sudo systemctl enable --now lt-ausys
journalctl -u lt-ausys -f        # «model ... loaded in 2.1 s», «listening on tcp://0.0.0.0:10302»
```

Проверка с любой машины, где установлен пакет `wyoming`:

```bash
python tools/wyoming_client_test.py --host <ip-сервера> --port 10302 komanda.wav
```

Для модели fp32 — `allow_patterns=['*.json','vocab.txt','encoder-model.onnx','encoder-model.onnx.data','decoder_joint-model.onnx']`
и `--quant fp32` в unit-файле. Для коротких команд `--threads 2` чуть быстрее, чем 4.

### C. Home Assistant

Настройки → Устройства и службы → **Добавить интеграцию → Wyoming Protocol** → host = сервер, port = `10302`.
Home Assistant покажет `lt_ausys` с версией модели. В конвейере Assist выберите его как движок **Speech-to-text** и
язык — литовский. Перед ним, как обычно, ставьте слово-активатор / VAD (спутник ESPHome,
[Speakerphone Assist](https://github.com/RobertasTa/speakerphone-assist) и т. п.): сервер распознаёт всё, что ему
присылают, а на чистом шуме придумывает текст, как любая модель распознавания речи.

### D. Уже работает `wyoming-faster-whisper`? Новый сервер не нужен

Начиная с версии 3.9 у [`wyoming-faster-whisper`](https://github.com/OHF-Voice/wyoming-faster-whisper) есть движок
`onnx-asr`, который загружает любую onnx-asr модель с Hugging Face. Ничего никуда сливать не нужно:

```bash
pip install "wyoming-faster-whisper[onnx_asr]"      # или добавьте extra к существующей установке
wyoming-faster-whisper --uri tcp://0.0.0.0:10302 --stt-library onnx-asr --model RobertasTa/lt-ausys \
    --language lt --data-dir /data --download-dir /data
```

Две оговорки, найденные в его коде (3.9.0): этот движок загружает файлы **fp32** (загрузка 2,5 ГБ) — выбор квантования
он не передаёт — и `--cpu-threads` к нему не применяется. Если нужны int8 и управление потоками — используйте
маленький сервер выше. Позволяет ли дополнение *Whisper* для Home Assistant OS задать `--stt-library`, мы не проверяли.

**Версии.** `--model` сервера — символическая ссылка; новая версия модели — новый каталог и `ln -sfn` + restart.
Откат — та же команда. Каждая фраза пишется в журнал (длина звука, время распознавания, текст), так что новую
версию можно наблюдать в журнале, пока ей не доверяешь.

## Конвертируйте свою модель Parakeet

Тот же путь подходит для любой NeMo Parakeet-TDT / RNNT модели (например, дообучения на другом языке):

```bash
# нужно окружение NeMo (nemo_toolkit[asr] 3.x) — хватает CPU, ~15 с для модели 0,6B
python tools/export_nemo_to_onnx.py --nemo model.nemo --out model_fp32
# нужны только onnxruntime + onnx
python tools/quantize_int8_matmul.py --model-dir model_fp32
# измерение (список = TSV путь<TAB>эталон)
python tools/bench.py --engine onnx-asr --model model_fp32 --quant int8 --list test.tsv --out results/
```

Две вещи, которые скрипты знают и которые стоили нам дня: NeMo записывает энкодер больше 2 ГБ сотнями файлов по
одному на тензор (скрипт экспорта упаковывает их в один `.onnx.data`), а **квантование свёрток вредит** — квантуются
только `MatMul`, что сохраняет точность fp32 и к тому же самый быстрый вариант ([почему](docs/results.md#conversion-checks)).

## Что планируется

* **v1** — та же модель, дообученная на домашних голосовых командах (один говорящий, конференц-спикерфон, радио на
  фоне), измеренная на том же стенде перед заменой v0.
* Использовать LT_ausys и на этапе слова-активатора, отправив на пенсию маленький Whisper, который сейчас слушает имя.

## Благодарности

* **Кристийонас Якубсонас** — [`kmynas-parakeet-lt-v4`](https://huggingface.co/kristijonas/kmynas-parakeet-lt-v4),
  сама модель: Parakeet-TDT 0.6B v3, дообученная на корпусе LIEPA-3, опубликованная под CC BY 4.0 с честной
  карточкой. Всё, что добавляет этот проект, — упаковка, сервер и измерения; уши — его.
* **NVIDIA** — [Parakeet-TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) (CC BY 4.0).
* **Вильнюсский университет / raštija.lt** — речевой корпус LIEPA-3 (CC BY 4.0).
* **Илья Ступаков** — [onnx-asr](https://github.com/istupakov/onnx-asr); **Microsoft** — onnxruntime;
  **Open Home Foundation** — [протокол Wyoming](https://github.com/OHF-Voice/wyoming).

## Лицензия

Код в этом репозитории: **GPL-3.0** ([LICENSE](LICENSE)). Файлы модели на Hugging Face: **CC BY 4.0**, как и у
оригинала — распространяя или изменяя их, укажите Кристийонаса Якубсонаса и скажите, что вы изменили.

---

Часть [Claude's Gifts to the World](https://github.com/RobertasTa) — бесплатные инструменты с открытым кодом,
которые ИИ создаёт вместе с одним человеком, проверяющим их дома.
