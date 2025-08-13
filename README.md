---

# 🎙️ SSL Anti-Spoofing

A **Self-Supervised Learning (SSL)** based pipeline for **audio anti-spoofing detection**.
This project extracts embeddings from powerful SSL models and classifies them into spoofed or bona fide speech.

---

## 📌 Overview

This pipeline consists of two main stages:

1. **Frontend – SSL Feature Extractors**
   Pretrained audio encoders (from [S3PRL](https://github.com/s3prl/s3prl)) such as:

   * `wavlm_large`
   * `mae_ast_frame`
   * `npc_960hr`
   * *(hundreds more available — see full list below)*

2. **Backend – Classifier Models**
   Models trained on extracted SSL embeddings:

   * `aasist`
   * `sls`
   * `xlsrmamba`

---

## ✨ Features

* **Multiple backend architectures**: AASIST, SLS, XLSR-Mamba.
* **Large variety of SSL frontends**: WavLM, MAE-AST, NPC, HuBERT, wav2vec 2.0, Data2Vec, etc.
* **Simple CLI & YAML config system** for quick model switching.
* **Training & evaluation pipelines** in `main.py` / `main2.py`.
* **Automatic checkpoint saving** and organized output directories.

---

## 📂 Repo Structure

```
.
├── main.py                 # Training & evaluation script
├── config.py               # Global config handling
├── models/                 # Classifier architectures
├── ssl_models/             # SSL feature extraction modules
├── protocols/              # Protocol files for datasets
├── requirements.txt        # Python dependencies
└── utils/                  # Helper functions
```

---

## 🛠 Supported Models

### Classifiers

```
aasist | sls | xlsrmamba
```

### SSL Feature Extractors

Includes common options like:

```
wavlm_large | mae_ast_frame | npc_960hr | hubert | wav2vec2_large_ll60k | data2vec_large_ll60k | tera
```

<details>
<summary>📜 Show All SSL Features</summary>

```
apc | apc_360hr | apc_960hr | apc_local | apc_url | ast | audio_albert | audio_albert_960hr | audio_albert_local | audio_albert_logMelBase_T_share_AdamW_b32_1m_960hr_drop1 | audio_albert_url | baseline | baseline_local | byol_a_1024 | byol_a_2048 | byol_a_512 | byol_s_cvt | byol_s_default | byol_s_resnetish34 | contentvec | contentvec_km100 | contentvec_km500 | cpc_local | cpc_url | customized_upstream | cvhubert | data2vec | data2vec_base_960 | data2vec_custom | data2vec_large_ll60k | data2vec_local | data2vec_url | decoar | decoar2 | decoar2_custom | decoar2_local | decoar2_url | decoar_custom | decoar_layers | decoar_layers_custom | decoar_layers_local | decoar_layers_url | decoar_local | decoar_url | discretebert | distilhubert | distilhubert_base | distiller_local | distiller_url | espnet_hubert_base_iter0 | espnet_hubert_base_iter1 | espnet_hubert_custom | espnet_hubert_large_gs_ll60k | espnet_hubert_local | fbank | fbank_no_cmvn | hf_hubert_custom | hf_wav2vec2_custom | hubert | hubert_base | hubert_base_robust_mgr | hubert_custom | hubert_large_ll60k | hubert_local | hubert_url | lighthubert | lighthubert_base | lighthubert_local | lighthubert_small | lighthubert_stage1 | lighthubert_url | linear | mae_ast_frame | mae_ast_local | mae_ast_patch | mae_ast_url | mel | mfcc | mhubert_base_vp_en_es_fr_it3 | mockingjay | mockingjay_100hr | mockingjay_960hr | mockingjay_local | mockingjay_logMelBase_T_AdamW_b32_1m_960hr | mockingjay_logMelBase_T_AdamW_b32_1m_960hr_drop1 | mockingjay_logMelBase_T_AdamW_b32_1m_960hr_seq3k | mockingjay_logMelBase_T_AdamW_b32_200k_100hr | mockingjay_logMelLinearLarge_T_AdamW_b32_500k_360hr_drop1 | mockingjay_origin | mockingjay_url | modified_cpc | mos_apc | mos_apc_local | mos_apc_url | mos_tera | mos_tera_local | mos_tera_url | mos_wav2vec2 | mos_wav2vec2_local | mos_wav2vec2_url | ms_hubert | multires_hubert_base | multires_hubert_custom | multires_hubert_large | multires_hubert_local | multires_hubert_multilingual_base | multires_hubert_multilingual_large400k | multires_hubert_multilingual_large600k | npc | npc_360hr | npc_960hr | npc_local | npc_url | pase_local | pase_plus | pase_url | passt_base | passt_base20sec | passt_base2level | passt_base2levelmel | passt_base30sec | passt_hop100base | passt_hop100base2lvl | passt_hop100base2lvlmel | passt_hop160base | passt_hop160base2lvl | passt_hop160base2lvlmel | spectrogram | ssast_frame_base | ssast_patch_base | stft_mag | tera | tera_100hr | tera_960hr | tera_fbankBase_T_F_AdamW_b32_200k_100hr | tera_local | tera_logMelBase_T_F_AdamW_b32_1m_960hr | tera_logMelBase_T_F_AdamW_b32_1m_960hr_drop1 | tera_logMelBase_T_F_AdamW_b32_1m_960hr_seq3k | tera_logMelBase_T_F_AdamW_b32_200k_100hr | tera_logMelBase_T_F_M_AdamW_b32_1m_960hr_drop1 | tera_logMelBase_T_F_M_AdamW_b32_200k_100hr | tera_url | timit_posteriorgram | unispeech_sat | unispeech_sat_base | unispeech_sat_base_plus | unispeech_sat_large | unispeech_sat_local | unispeech_sat_url | vggish | vq_apc | vq_apc_360hr | vq_apc_960hr | vq_apc_url | vq_wav2vec | vq_wav2vec_custom | vq_wav2vec_gumbel | vq_wav2vec_kmeans | vq_wav2vec_kmeans_roberta | wav2vec | wav2vec2 | wav2vec2_base_960 | wav2vec2_base_s2st_en_librilight | wav2vec2_base_s2st_es_voxpopuli | wav2vec2_conformer_large_s2st_en_librilight | wav2vec2_conformer_large_s2st_es_voxpopuli | wav2vec2_conformer_relpos | wav2vec2_conformer_rope | wav2vec2_custom | wav2vec2_large_960 | wav2vec2_large_ll60k | wav2vec2_large_lv60_cv_swbd_fsh | wav2vec2_large_voxpopuli_100k | wav2vec2_local | wav2vec2_url | wav2vec_custom | wav2vec_large | wav2vec_local | wav2vec_url | wavlablm_ek_40k | wavlablm_mk_40k | wavlablm_ms_40k | wavlm | wavlm_base | wavlm_base_plus | wavlm_large | wavlm_local | wavlm_url | xls_r_1b | xls_r_2b | xls_r_300m | xlsr_53
```

</details>


## ⚡ Quick Start

### 1️⃣ Installation

```bash
git clone <repo_url>
cd ssl-antispoofing
pip install -r requirements.txt
```

### 2️⃣ Configure

Edit `config.yampyl`:

```config.py
ssl_feature: wavlm_large
model_arch: aasist
mode: train
save_dir: output/models
batch_size: 14
learning_rate: 1e-6
num_epochs: 50
```

Or override via CLI:

```bash
# Example command
python main.py \
  --batch_size 14 \
  --num_epochs 50 \
  --lr 1e-6 \
  --weight_decay 1e-4 \
  --ssl_feature wavlm_large \
  --seed 1234 \
  --emb_size 256 \
  --num_encoders 12
```

### 3️⃣ Train

```bash
python main.py --batch_size 14 --num_epochs 50 --lr 1e-6 --weight_decay 1e-4 --ssl_feature wavlm_large --seed 1234 --emb_size 256 --num_encoders 12
```

### 4️⃣ Evaluate

#### Set the evaluation dataset in config.py at dev 
```bash
python main.py --model_path output/models/your_model.pth
```

---

## 🔄 Switching Components

**Change SSL Extractor**


```bash
--ssl_feature mae_ast_frame
```
---

## 📂 Outputs

* **Checkpoints** → stored in `save_dir`
* **Logs** → stored during training/evaluation

---

## 📚 References

* [S3PRL](https://github.com/s3prl/s3prl)
* WavLM, HuBERT, wav2vec 2.0, Data2Vec, BYOL-A, TERA, Mockingjay, SSAST, PaSST, UniSpeech-SAT, XLS-R, DeCoAR, PASE+, VQ-wav2vec, ContentVec, LightHuBERT, DistilHuBERT, etc.

---

