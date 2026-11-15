# LT_ausys — lietuviškos ausys Home Assistant'ui

![LT_ausys — robotinė ranka laiko dovanų dėžutę su užrašu lt-ausys po švytinčiu gaubliu; kampe LIEPA-3 garsyno įstaigų ženklai (iliustracija sukurta naudojant DI)](docs/banner.png)

**Lietuviškas kalbos atpažinimas, kuris sukasi ant mažo procesoriaus pakankamai greitai balso asistentui: 0,2 s
komandai vietoj 3,5 s.** Lietuviškai išmokytas transduserio modelis ([Upelis](https://huggingface.co/datolabs/upelis-0.6b-lt-v1),
Datolabs), perkeltas į ONNX ir suspaustas, paduodamas per Wyoming protokolą — be vaizdo plokštės, be PyTorch, niekas
neišeina iš namų.

🌐 [English](README.md) · **Lietuvių** · [Русский](README_ru.md)

> ℹ️ Sukurta naudojant DI: konversiją, serverį, matavimus ir šiuos tekstus darė Claude (Anthropic) kartu su Robertu,
> išbandyta su tikra įranga. Pats modelis — Datolabs Upelis (nuo 2026-11-15; iki tol buvo Kristijono Jakubsono kmynas),
> žr. [Padėkos](#padėkos).

> Išmatuota 2026-11-15 ant Ryzen 7 8845HS mini PC (4 gijos, Proxmox LXC): 2–6 s balso komanda atpažįstama per
> **0,18 s** (Whisper large-v3-turbo / Paprika tame pačiame procesoriuje: 3,55 s), FLEURS lietuviška WER **8,3 %**
> (Paprika 12,0 %). Ankstesnė versija nuo 2026-10-24 — namų balso asistento ausys.

---

## Ką gauni

| | |
|---|---|
| **Modelis** | [`RobertasTa/lt-ausys`](https://huggingface.co/RobertasTa/lt-ausys) Hugging Face'e — fp32 (2,5 GB) ir int8 (0,7 GB), [onnx-asr](https://github.com/istupakov/onnx-asr) katalogo formatu. OpenMDW-1.1 (laikyk `LICENSE` ir `NOTICE`). |
| **Serveris** | [`server/lt_ausys_server.py`](server/lt_ausys_server.py) — Wyoming kalbos atpažinimo serveris (~120 eil.) + systemd unit'as. Home Assistant jį randa kaip Wyoming integraciją. |
| **Receptas** | [`tools/`](tools/) — NeMo transduserio modelį į ONNX (Parakeet arba modelį su kalbos užuomina, kaip Upelis), suspausti taip, kad nekenktų, išmatuoti. |
| **Skaičiai** | [`docs/results.md`](docs/results.md) — trys testų rinkiniai, keturi modeliai, viena mašina. |

Išvestis — normali rašytinė lietuvių kalba: skyryba, didžiosios; skaičiai nuo 10 — skaitmenimis, 1–9 — žodžiais
(*„Įjunk šviesą svetainėje 20 procentų."*).

> ⚠️ **Diktavimui imkite ankstesnę versiją.** Šios ausys retai rašo taškus: ~45 s ištisinėje kalboje jos rado 2,8 %
> sakinių ribų (kmynas — 84,7 %) ir maždaug pusę kablelių. Komandoms tai nesvarbu; tekstui, kurį žmonės skaito, imkite
> Hugging Face reviziją `v1-kmynas` — taip daro ir mūsų diktavimo programa
> [Diktuoklė](https://github.com/RobertasTa/diktuokle) ([skaičiai](docs/results.md)).

## Kodėl ne Whisper

Whisper šeimos modeliai kiekvienai frazei, kad ir kokiai trumpai, koduoja fiksuotą **30 sekundžių langą**. Ant
procesoriaus tai pastovi kelių sekundžių kaina komandai, ir joks suspaudimas jos nepanaikina. Transduseris
(RNN-T / TDT) apdoroja tik tiek garso, kiek jam paduota. Ta pati mašina, tos pačios 127 namų komandos:

| | komandai | FLEURS WER | komandų ketinimo egzaminas |
|---|---|---|---|
| **LT_ausys int8** | **0,18 s** | **8,3 %** | **103 / 119** |
| LT_ausys iki 2026-11-15 (kmynas, int8) | 0,21 s | 10,6 % | 101 / 119 |
| Paprika (Whisper large-v3-turbo, CT2 int8) | 3,55 s | 12,0 % | 93 / 119 |
| faster-whisper tiny int8 | 0,73 s | 99 % | 17 / 119 |

Pilnos lentelės, rinkiniai ir išlygos: [docs/results.md](docs/results.md).

## Nuo nulio: lietuviškas balso asistentas namuose

**Kam to reikia.** Iš dėžutės Home Assistant siūlo Whisper *base* (numatytas procesoriui) arba *small* (numatytas
vaizdo plokštei) — lietuviškoje skaitomoje kalboje jie daro **88 % ir 66 % žodžių klaidų** (mūsų matavimas
[wyoming-faster-whisper PR #122](https://github.com/OHF-Voice/wyoming-faster-whisper/pull/122)): grandinė žalia,
sakiniai atrodo kaip sakiniai, o namai daro ne tą. Kita išeitis — debesies paslauga, kuri girdi viską, kas pasakoma
namuose. LT_ausys tame pačiame teste — **8,3 %**, lokaliai ir greičiau už abu.

**Ko reikia lietuviškai balso grandinei, diegimo eilės tvarka:**

1. **Ausys — šis projektas.** Mažas Linux (mini PC, Proxmox konteineris, turėtų tikti ir Pi 5 — 4 GB RAM) su serveriu
   iš B skyriaus arba tavo esamas `wyoming-faster-whisper` iš D skyriaus. Penkios minutės.
2. **Home Assistant** — pridėk kaip Wyoming integraciją (C skyrius) ir pasirink Assist pipeline'o kalbos atpažinimu,
   kalba lietuvių. Paties Home Assistant lietuviškos komandos (*„įjunk šviesą svetainėje"*) jau yra.
3. **Burna** — lietuviškas Piper balsas: [Reginutė](https://github.com/RobertasTa/reginute) arba
   [Ingutė](https://github.com/RobertasTa/ingute). **Šiandien:** leisk jį per mažą Wyoming serverį iš balso
   repozitorijos (skyrius apie Home Assistant, penkios minutės) ir pridėk kaip Wyoming teksto į kalbą integraciją
   lygiai taip pat. Abu balsai jau yra oficialiame Piper kataloge, bet katalogo įrašams reikia Piper 1.9.0, kuris dar
   neišleistas. **Kai išeis Piper 1.9.0**, tikimės, kad veiks tiesiai iš Home Assistant Piper priedo — dar
   nepatikrinta; patvirtinsim čia.
4. **Mikrofonas kambaryje** — Home Assistant Voice PE, bet koks ESPHome palydovas arba USB konferencinė kolonėlė per
   [Speakerphone Assist](https://github.com/RobertasTa/speakerphone-assist). Tai, kas klauso žadinimo žodžio ir siunčia
   tavo sakinį ausims.
5. Pasakyk *„Įjunk šviesą svetainėje"* ir žiūrėk į lempą.

Viskas, kas aukščiau, sukasi tavo namuose; niekur nieko nesiunčiama.

## Greita pradžia

### A. Tik Python

```bash
pip install "onnx-asr[cpu,hub]"
```

```python
import onnx_asr
model = onnx_asr.load_model("RobertasTa/lt-ausys", quantization="int8")   # be quantization= — fp32
print(model.recognize("komanda.wav"))      # 16 kHz mono 16 bitų WAV arba float32 numpy masyvas su sample_rate=
```

### B. Wyoming serveris (bet koks Linux; išbandyta Proxmox LXC konteineryje)

Užtenka konteinerio ar VM su **4 branduoliais ir 4 GB RAM** (int8 viršūnė ~1,4 GB komandoms, ~1,7 GB ilgiems įrašams).

```bash
# 1. aplinka
sudo useradd -r -d /opt/lt_ausys -s /usr/sbin/nologin ltausys
sudo apt install -y python3-venv git          # Debian / Ubuntu
sudo mkdir -p /opt/lt_ausys && sudo git clone https://github.com/RobertasTa/lt-ausys /opt/lt_ausys/src
cd /opt/lt_ausys
sudo python3 -m venv venv && sudo venv/bin/pip install "onnx-asr[cpu,hub]" numpy wyoming

# 2. modelis (tik int8 failai — 0,7 GB)
sudo venv/bin/python3 -c "from huggingface_hub import snapshot_download as s; s('RobertasTa/lt-ausys', local_dir='model_v0', allow_patterns=['*.json','vocab.txt','*.int8.onnx'])"
sudo ln -sfn model_v0 model

# 3. serveris
sudo cp src/server/lt_ausys_server.py . && sudo cp src/server/lt-ausys.service /etc/systemd/system/
sudo chown -R ltausys:ltausys /opt/lt_ausys
sudo systemctl daemon-reload && sudo systemctl enable --now lt-ausys
journalctl -u lt-ausys -f        # „model ... loaded in 2.1 s", „listening on tcp://0.0.0.0:10302"
```

Patikrinimas iš bet kurios mašinos, kurioje įdiegtas `wyoming` paketas:

```bash
python tools/wyoming_client_test.py --host <serverio-ip> --port 10302 komanda.wav
```

Bandymo klientas priima 16 kHz mono 16 bitų WAV — įrašyk `arecord -f S16_LE -r 16000 -c 1 -d 4 komanda.wav`
arba pasiversk `ffmpeg -i bet_kas.wav -ar 16000 -ac 1 -sample_fmt s16 komanda.wav`; su float WAV jis sustoja
klaida `unknown format: 3`.

fp32 modeliui — `allow_patterns=['*.json','vocab.txt','encoder-model.onnx','encoder-model.onnx.data','decoder_joint-model.onnx']`
ir `--quant fp32` unit'e. Trumpoms komandoms `--threads 2` šiek tiek greičiau nei 4.

**Arba Docker** (išbandyta 2026-10-24 švariame Ubuntu 24.04 konteineryje: vaizdas 2,25 GB, 1,4 GiB RAM po 9,5 s
sakinio, kuris grįžo pažodžiui per 0,61 s):

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

Nustatymai → Įrenginiai ir paslaugos → **Pridėti integraciją → Wyoming Protocol** → host = serveris, port = `10302`.
Home Assistant parodo `lt_ausys` su modelio versija. Assist pipeline'e pasirink jį kaip **Speech-to-text** variklį ir
kalbą — lietuvių. Prieš jį, kaip įprasta, statyk žadinimo žodį / VAD (ESPHome palydovą,
[Speakerphone Assist](https://github.com/RobertasTa/speakerphone-assist) ir pan.): serveris atpažįsta viską, ką gauna,
o iš gryno triukšmo prigalvoja teksto, kaip ir kiekvienas kalbos atpažinimo modelis.

### D. Jau sukasi `wyoming-faster-whisper`? Naujo serverio nereikia

Nuo 3.9 versijos [`wyoming-faster-whisper`](https://github.com/OHF-Voice/wyoming-faster-whisper) turi `onnx-asr`
variklį, kuris krauna bet kurį onnx-asr modelį iš Hugging Face. Nieko niekur sulieti nereikia:

```bash
pip install "wyoming-faster-whisper[onnx_asr]"      # arba pridėk extra prie esamo diegimo
wyoming-faster-whisper --uri tcp://0.0.0.0:10302 --stt-library onnx-asr --model RobertasTa/lt-ausys \
    --language lt --data-dir /data --download-dir /data
```

Dvi išlygos, rastos jo kode (3.9.0): tas variklis krauna **fp32** failus (2,5 GB parsisiuntimas) — kvantavimo pasirinkimo
jis neperduoda — ir `--cpu-threads` jam negalioja. Jei nori int8 ir gijų valdymo — naudok mažąjį serverį aukščiau.
Ar Home Assistant OS *Whisper* priedas leidžia `--stt-library`, netikrinome.

**Versijos.** Serverio `--model` yra simbolinė nuoroda; nauja modelio versija — naujas katalogas ir `ln -sfn` +
restart. Grįžimas — ta pati komanda. Kiekviena frazė rašoma į žurnalą (garso trukmė, atpažinimo laikas, tekstas), tad
naują versiją galima stebėti žurnale, kol ja pasitiki.

## Paversk savo Parakeet modelį

Tas pats kelias tinka bet kuriam NeMo Parakeet-TDT / RNNT modeliui (pvz., kitos kalbos damokymui):

```bash
# reikia NeMo aplinkos (nemo_toolkit[asr] 3.x) — užtenka CPU, ~15 s 0,6B modeliui
python tools/export_nemo_to_onnx.py --nemo model.nemo --out model_fp32
# reikia tik onnxruntime + onnx
python tools/quantize_int8_matmul.py --model-dir model_fp32
# matavimas (sąrašas = TSV kelias<TAB>etalonas)
python tools/bench.py --engine onnx-asr --model model_fp32 --quant int8 --list test.tsv --out results/
```

**Modeliai su kalbos užuomina** (NeMo unified ASR, pvz. Upelis — toks dabar yra LT_ausys): vietoj pirmo žingsnio
`tools/export_nemo_prompt_to_onnx.py --nemo model.nemo --lang lt --out model_fp32 --check a.wav b.wav`. Paprastas
`model.export()` užuominą praranda, nes ji taikoma po enkoderio; skriptas vieną kalbą įsiuva į enkoderį ir prieš
eksportą patikrina, ar rezultatas sutampa su originalu.

Du dalykai, kuriuos skriptai žino ir kurie mums kainavo dieną: NeMo >2 GB enkoderį išrašo šimtais failų po tenzorių
(eksporto skriptas supakuoja juos į vieną `.onnx.data`), o **konvoliucijų kvantavimas kenkia** — kvantuojami tik
`MatMul`, ir tai išlaiko tikslumą lygų fp32 bei yra greičiausias variantas ([kodėl](docs/results.md#conversion-checks)).

## Kas planuojama

* LT_ausys ir žadinimo žodžio pakopai — išleidžiant į pensiją mažąjį Whisper, kuris dabar klauso vardo.

**Istorija.** Iki 2026-11-15 LT_ausys buvo Kristijono Jakubsono kmynas v4 (Parakeet-TDT) ONNX pavidalu — jis tebėra
pasiekiamas: Hugging Face versija `v1-kmynas`, šio repo žymė `v1.0-kmynas`. Pakeistas, nes Upelis tose pačiose namų
komandose ir tame pačiame procesoriuje buvo tikslesnis ir naudojo mažiau atminties ([skaičiai](docs/results.md)).

## Padėkos

* **Datolabs** — [`upelis-0.6b-lt-v1`](https://huggingface.co/datolabs/upelis-0.6b-lt-v1), pats modelis: NVIDIA
  Nemotron unified ASR 0.6B, damokytas LIEPA-3 ir Common Voice, paskelbtas OpenMDW-1.1. Viskas, ką šis projektas
  prideda, yra ONNX eksportas (su įsiūta lietuvių kalbos užuomina), suspaudimas, serveris ir matavimai; ausys — jų.
* **NVIDIA** — [Nemotron unified ASR 0.6B](https://huggingface.co/nvidia/nemotron-labs-unified-asr-0.6b) (OpenMDW-1.1).
* **Vytauto Didžiojo universitetas, Vilniaus universitetas ir Lietuvių kalbos institutas** — LIEPA-3 garsynas
  (CC BY 4.0); **Mozilla** — Common Voice (CC0).
* **Kristijonas Jakubsonas** — [`kmynas-parakeet-lt-v4`](https://huggingface.co/kristijonas/kmynas-parakeet-lt-v4),
  kuris buvo LT_ausys iki 2026-11-15 (CC BY 4.0).
* **Ilya Stupakov** — [onnx-asr](https://github.com/istupakov/onnx-asr); **Microsoft** — onnxruntime;
  **Open Home Foundation** — [Wyoming protokolas](https://github.com/OHF-Voice/wyoming).

## Licencija

Šio repo kodas: **GPL-3.0** ([LICENSE](LICENSE)). Modelio failai Hugging Face'e: **OpenMDW-1.1**, kaip ir originalo —
platindamas palik kartu esančius `LICENSE` ir `NOTICE` failus.

---

Dalis [Claude's Gifts to the World](https://github.com/RobertasTa) — nemokami atviro kodo įrankiai, kuriuos DI kuria
kartu su vienu žmogumi, išbandančiu juos namuose.
