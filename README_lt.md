# LT_ausys — lietuviškos ausys Home Assistant'ui

![LT_ausys — robotinė ranka laiko dovanų dėžutę su užrašu lt-ausys po švytinčiu gaubliu; kampe LIEPA-3 garsyno įstaigų ženklai (iliustracija sukurta naudojant DI)](docs/banner.png)

**Lietuviškas kalbos atpažinimas, kuris sukasi ant mažo procesoriaus pakankamai greitai balso asistentui: 0,2 s
komandai vietoj 3,5 s.** Parakeet-TDT modelis, apmokytas 3 087 valandomis lietuviškos kalbos, perkeltas į ONNX ir
suspaustas, paduodamas per Wyoming protokolą — be vaizdo plokštės, be PyTorch, niekas neišeina iš namų.

🌐 [English](README.md) · **Lietuvių** · [Русский](README_ru.md)

> ℹ️ Sukurta naudojant DI: konversiją, serverį, matavimus ir šiuos tekstus darė Claude (Anthropic) kartu su Robertu,
> išbandyta su tikra įranga. Pats modelis — Kristijono Jakubsono, žr. [Padėkos](#padėkos).

> Išmatuota 2026-10-24 ant Ryzen 7 8845HS mini PC (4 gijos, Proxmox LXC): 2–6 s balso komanda atpažįstama per
> **0,21 s** (Whisper large-v3-turbo / Paprika tame pačiame procesoriuje: 3,55 s), FLEURS lietuviška WER **10,6 %**
> (Paprika 12,0 %). Nuo to vakaro — namų balso asistento ausys.

---

## Ką gauni

| | |
|---|---|
| **Modelis** | [`RobertasTa/lt-ausys`](https://huggingface.co/RobertasTa/lt-ausys) Hugging Face'e — fp32 (2,5 GB) ir int8 (0,9 GB), [onnx-asr](https://github.com/istupakov/onnx-asr) katalogo formatu. CC BY 4.0. |
| **Serveris** | [`server/lt_ausys_server.py`](server/lt_ausys_server.py) — Wyoming kalbos atpažinimo serveris (~120 eil.) + systemd unit'as. Home Assistant jį randa kaip Wyoming integraciją. |
| **Receptas** | [`tools/`](tools/) — bet kurį NeMo Parakeet modelį į ONNX, suspausti taip, kad nekenktų, išmatuoti. |
| **Skaičiai** | [`docs/results.md`](docs/results.md) — trys testų rinkiniai, keturi modeliai, viena mašina. |

Išvestis — normali rašytinė lietuvių kalba: skyryba, didžiosios, skaičiai žodžiais (*„Įjunk šviesą svetainėje
dvidešimt procentų."*).

## Kodėl ne Whisper

Whisper šeimos modeliai kiekvienai frazei, kad ir kokiai trumpai, koduoja fiksuotą **30 sekundžių langą**. Ant
procesoriaus tai pastovi kelių sekundžių kaina komandai, ir joks suspaudimas jos nepanaikina. Transduseris
(Parakeet-TDT) apdoroja tik tiek garso, kiek jam paduota. Ta pati mašina, tos pačios 127 namų komandos:

| | komandai | FLEURS WER | komandų ketinimo egzaminas |
|---|---|---|---|
| **LT_ausys v0 int8** | **0,21 s** | **10,6 %** | **101 / 119** |
| Paprika (Whisper large-v3-turbo, CT2 int8) | 3,55 s | 12,0 % | 93 / 119 |
| faster-whisper tiny int8 | 0,73 s | 99 % | 17 / 119 |

Pilnos lentelės, rinkiniai ir išlygos: [docs/results.md](docs/results.md).

## Nuo nulio: lietuviškas balso asistentas namuose

**Kam to reikia.** Iš dėžutės Home Assistant siūlo Whisper *base* (numatytas procesoriui) arba *small* (numatytas
vaizdo plokštei) — lietuviškoje skaitomoje kalboje jie daro **88 % ir 66 % žodžių klaidų** (mūsų matavimas
[wyoming-faster-whisper PR #122](https://github.com/OHF-Voice/wyoming-faster-whisper/pull/122)): grandinė žalia,
sakiniai atrodo kaip sakiniai, o namai daro ne tą. Kita išeitis — debesies paslauga, kuri girdi viską, kas pasakoma
namuose. LT_ausys tame pačiame teste — **10,6 %**, lokaliai ir greičiau už abu.

**Ko reikia lietuviškai balso grandinei, diegimo eilės tvarka:**

1. **Ausys — šis projektas.** Mažas Linux (mini PC, Proxmox konteineris, turėtų tikti ir Pi 5 — 4 GB RAM) su serveriu
   iš B skyriaus arba tavo esamas `wyoming-faster-whisper` iš D skyriaus. Penkios minutės.
2. **Home Assistant** — pridėk kaip Wyoming integraciją (C skyrius) ir pasirink Assist pipeline'o kalbos atpažinimu,
   kalba lietuvių. Paties Home Assistant lietuviškos komandos (*„įjunk šviesą svetainėje"*) jau yra.
3. **Burna** — lietuviškas balsas Piper'iui: [Reginutė](https://github.com/RobertasTa/reginute) arba
   [Ingutė](https://github.com/RobertasTa/ingute), abi oficialiame Piper balsų kataloge; Piper pridedamas kaip Wyoming
   teksto į kalbą integracija lygiai taip pat.
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

Užtenka konteinerio ar VM su **4 branduoliais ir 4 GB RAM** (int8 viršūnė ~1,6 GB komandoms, ~2,3 GB ilgiems įrašams).

```bash
# 1. aplinka
sudo useradd -r -d /opt/lt_ausys -s /usr/sbin/nologin ltausys
sudo mkdir -p /opt/lt_ausys && cd /opt/lt_ausys
sudo python3 -m venv venv && sudo venv/bin/pip install "onnx-asr[cpu,hub]" numpy wyoming

# 2. modelis (tik int8 failai — 0,9 GB)
sudo venv/bin/python3 -c "from huggingface_hub import snapshot_download as s; s('RobertasTa/lt-ausys', local_dir='model_v0', allow_patterns=['*.json','vocab.txt','*.int8.onnx'])"
sudo ln -sfn model_v0 model

# 3. serveris
sudo cp server/lt_ausys_server.py . && sudo cp server/lt-ausys.service /etc/systemd/system/
sudo chown -R ltausys:ltausys /opt/lt_ausys
sudo systemctl daemon-reload && sudo systemctl enable --now lt-ausys
journalctl -u lt-ausys -f        # „model ... loaded in 2.1 s", „listening on tcp://0.0.0.0:10302"
```

Patikrinimas iš bet kurios mašinos, kurioje įdiegtas `wyoming` paketas:

```bash
python tools/wyoming_client_test.py --host <serverio-ip> --port 10302 komanda.wav
```

fp32 modeliui — `allow_patterns=['*.json','vocab.txt','encoder-model.onnx','encoder-model.onnx.data','decoder_joint-model.onnx']`
ir `--quant fp32` unit'e. Trumpoms komandoms `--threads 2` šiek tiek greičiau nei 4.

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

Du dalykai, kuriuos skriptai žino ir kurie mums kainavo dieną: NeMo >2 GB enkoderį išrašo šimtais failų po tenzorių
(eksporto skriptas supakuoja juos į vieną `.onnx.data`), o **konvoliucijų kvantavimas kenkia** — kvantuojami tik
`MatMul`, ir tai išlaiko tikslumą lygų fp32 bei yra greičiausias variantas ([kodėl](docs/results.md#conversion-checks)).

## Kas planuojama

* **v1** — tas pats modelis, damokytas namų balso komandomis (vienas kalbėtojas, konferencinė kolonėlė, radijas
  fone), išmatuotas tuo pačiu stendu, prieš pakeičiant v0.
* LT_ausys ir žadinimo žodžio pakopai — išleidžiant į pensiją mažąjį Whisper, kuris dabar klauso vardo.

## Padėkos

* **Kristijonas Jakubsonas** — [`kmynas-parakeet-lt-v4`](https://huggingface.co/kristijonas/kmynas-parakeet-lt-v4),
  pats modelis: Parakeet-TDT 0.6B v3, damokytas LIEPA-3 garsynu, paskelbtas CC BY 4.0 su sąžininga kortele. Viskas,
  ką šis projektas prideda, yra pakavimas, serveris ir matavimai; ausys — jo.
* **NVIDIA** — [Parakeet-TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) (CC BY 4.0).
* **Vilniaus universitetas / raštija.lt** — LIEPA-3 garsynas (CC BY 4.0).
* **Ilya Stupakov** — [onnx-asr](https://github.com/istupakov/onnx-asr); **Microsoft** — onnxruntime;
  **Open Home Foundation** — [Wyoming protokolas](https://github.com/OHF-Voice/wyoming).

## Licencija

Šio repo kodas: **GPL-3.0** ([LICENSE](LICENSE)). Modelio failai Hugging Face'e: **CC BY 4.0**, kaip ir originalo —
platindamas ar keisdamas nurodyk Kristijoną Jakubsoną ir pasakyk, ką pakeitei.

---

Dalis [Claude's Gifts to the World](https://github.com/RobertasTa) — nemokami atviro kodo įrankiai, kuriuos DI kuria
kartu su vienu žmogumi, išbandančiu juos namuose.
