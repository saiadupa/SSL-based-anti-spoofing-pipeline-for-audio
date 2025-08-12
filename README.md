---

# SSL Anti-Spoofing

This repository implements a self-supervised learning (SSL) based anti-spoofing pipeline for audio. It has two main stages:

 **Frontend (SSL feature extractors):** wavlm\_large, mae\_ast\_frame, npc\_960hr
 **Backend (Classifier models):** AASIST, SLS, XLSR-Mamba

---

## Features

<table>
  <tr>
    <th align="left">Classifier models</th>
  </tr>
  <tr>
    <td><code>aasist</code> &nbsp;|&nbsp; <code>sls</code> &nbsp;|&nbsp; <code>xlsrmamba</code></td>
  </tr>
</table>

<br/>

<table>
  <tr>
    <th align="left">SSL feature extractors (ready-to-use presets)</th>
  </tr>
  <tr>
  <td><code>apc</code> &nbsp;|&nbsp; <code>apc_360hr</code> &nbsp;|&nbsp; <code>apc_960hr</code> &nbsp;|&nbsp; <code>apc_local</code> &nbsp;|&nbsp; <code>apc_url</code> &nbsp;|&nbsp; <code>ast</code> &nbsp;|&nbsp; <code>audio_albert</code> &nbsp;|&nbsp; <code>audio_albert_960hr</code> &nbsp;|&nbsp; <code>audio_albert_local</code> &nbsp;|&nbsp; <code>audio_albert_logMelBase_T_share_AdamW_b32_1m_960hr_drop1</code> &nbsp;|&nbsp; <code>audio_albert_url</code> &nbsp;|&nbsp; <code>baseline</code> &nbsp;|&nbsp; <code>baseline_local</code> &nbsp;|&nbsp; <code>byol_a_1024</code> &nbsp;|&nbsp; <code>byol_a_2048</code> &nbsp;|&nbsp; <code>byol_a_512</code> &nbsp;|&nbsp; <code>byol_s_cvt</code> &nbsp;|&nbsp; <code>byol_s_default</code> &nbsp;|&nbsp; <code>byol_s_resnetish34</code> &nbsp;|&nbsp; <code>contentvec</code> &nbsp;|&nbsp; <code>contentvec_km100</code> &nbsp;|&nbsp; <code>contentvec_km500</code> &nbsp;|&nbsp; <code>cpc_local</code> &nbsp;|&nbsp; <code>cpc_url</code> &nbsp;|&nbsp; <code>customized_upstream</code> &nbsp;|&nbsp; <code>cvhubert</code> &nbsp;|&nbsp; <code>data2vec</code> &nbsp;|&nbsp; <code>data2vec_base_960</code> &nbsp;|&nbsp; <code>data2vec_custom</code> &nbsp;|&nbsp; <code>data2vec_large_ll60k</code> &nbsp;|&nbsp; <code>data2vec_local</code> &nbsp;|&nbsp; <code>data2vec_url</code> &nbsp;|&nbsp; <code>decoar</code> &nbsp;|&nbsp; <code>decoar2</code> &nbsp;|&nbsp; <code>decoar2_custom</code> &nbsp;|&nbsp; <code>decoar2_local</code> &nbsp;|&nbsp; <code>decoar2_url</code> &nbsp;|&nbsp; <code>decoar_custom</code> &nbsp;|&nbsp; <code>decoar_layers</code> &nbsp;|&nbsp; <code>decoar_layers_custom</code> &nbsp;|&nbsp; <code>decoar_layers_local</code> &nbsp;|&nbsp; <code>decoar_layers_url</code> &nbsp;|&nbsp; <code>decoar_local</code> &nbsp;|&nbsp; <code>decoar_url</code> &nbsp;|&nbsp; <code>discretebert</code> &nbsp;|&nbsp; <code>distilhubert</code> &nbsp;|&nbsp; <code>distilhubert_base</code> &nbsp;|&nbsp; <code>distiller_local</code> &nbsp;|&nbsp; <code>distiller_url</code> &nbsp;|&nbsp; <code>espnet_hubert_base_iter0</code> &nbsp;|&nbsp; <code>espnet_hubert_base_iter1</code> &nbsp;|&nbsp; <code>espnet_hubert_custom</code> &nbsp;|&nbsp; <code>espnet_hubert_large_gs_ll60k</code> &nbsp;|&nbsp; <code>espnet_hubert_local</code> &nbsp;|&nbsp; <code>fbank</code> &nbsp;|&nbsp; <code>fbank_no_cmvn</code> &nbsp;|&nbsp; <code>hf_hubert_custom</code> &nbsp;|&nbsp; <code>hf_wav2vec2_custom</code> &nbsp;|&nbsp; <code>hubert</code> &nbsp;|&nbsp; <code>hubert_base</code> &nbsp;|&nbsp; <code>hubert_base_robust_mgr</code> &nbsp;|&nbsp; <code>hubert_custom</code> &nbsp;|&nbsp; <code>hubert_large_ll60k</code> &nbsp;|&nbsp; <code>hubert_local</code> &nbsp;|&nbsp; <code>hubert_url</code> &nbsp;|&nbsp; <code>lighthubert</code> &nbsp;|&nbsp; <code>lighthubert_base</code> &nbsp;|&nbsp; <code>lighthubert_local</code> &nbsp;|&nbsp; <code>lighthubert_small</code> &nbsp;|&nbsp; <code>lighthubert_stage1</code> &nbsp;|&nbsp; <code>lighthubert_url</code> &nbsp;|&nbsp; <code>linear</code> &nbsp;|&nbsp; <code>mae_ast_frame</code> &nbsp;|&nbsp; <code>mae_ast_local</code> &nbsp;|&nbsp; <code>mae_ast_patch</code> &nbsp;|&nbsp; <code>mae_ast_url</code> &nbsp;|&nbsp; <code>mel</code> &nbsp;|&nbsp; <code>mfcc</code> &nbsp;|&nbsp; <code>mhubert_base_vp_en_es_fr_it3</code> &nbsp;|&nbsp; <code>mockingjay</code> &nbsp;|&nbsp; <code>mockingjay_100hr</code> &nbsp;|&nbsp; <code>mockingjay_960hr</code> &nbsp;|&nbsp; <code>mockingjay_local</code> &nbsp;|&nbsp; <code>mockingjay_logMelBase_T_AdamW_b32_1m_960hr</code> &nbsp;|&nbsp; <code>mockingjay_logMelBase_T_AdamW_b32_1m_960hr_drop1</code> &nbsp;|&nbsp; <code>mockingjay_logMelBase_T_AdamW_b32_1m_960hr_seq3k</code> &nbsp;|&nbsp; <code>mockingjay_logMelBase_T_AdamW_b32_200k_100hr</code> &nbsp;|&nbsp; <code>mockingjay_logMelLinearLarge_T_AdamW_b32_500k_360hr_drop1</code> &nbsp;|&nbsp; <code>mockingjay_origin</code> &nbsp;|&nbsp; <code>mockingjay_url</code> &nbsp;|&nbsp; <code>modified_cpc</code> &nbsp;|&nbsp; <code>mos_apc</code> &nbsp;|&nbsp; <code>mos_apc_local</code> &nbsp;|&nbsp; <code>mos_apc_url</code> &nbsp;|&nbsp; <code>mos_tera</code> &nbsp;|&nbsp; <code>mos_tera_local</code> &nbsp;|&nbsp; <code>mos_tera_url</code> &nbsp;|&nbsp; <code>mos_wav2vec2</code> &nbsp;|&nbsp; <code>mos_wav2vec2_local</code> &nbsp;|&nbsp; <code>mos_wav2vec2_url</code> &nbsp;|&nbsp; <code>ms_hubert</code> &nbsp;|&nbsp; <code>multires_hubert_base</code> &nbsp;|&nbsp; <code>multires_hubert_custom</code> &nbsp;|&nbsp; <code>multires_hubert_large</code> &nbsp;|&nbsp; <code>multires_hubert_local</code> &nbsp;|&nbsp; <code>multires_hubert_multilingual_base</code> &nbsp;|&nbsp; <code>multires_hubert_multilingual_large400k</code> &nbsp;|&nbsp; <code>multires_hubert_multilingual_large600k</code> &nbsp;|&nbsp; <code>npc</code> &nbsp;|&nbsp; <code>npc_360hr</code> &nbsp;|&nbsp; <code>npc_960hr</code> &nbsp;|&nbsp; <code>npc_local</code> &nbsp;|&nbsp; <code>npc_url</code> &nbsp;|&nbsp; <code>pase_local</code> &nbsp;|&nbsp; <code>pase_plus</code> &nbsp;|&nbsp; <code>pase_url</code> &nbsp;|&nbsp; <code>passt_base</code> &nbsp;|&nbsp; <code>passt_base20sec</code> &nbsp;|&nbsp; <code>passt_base2level</code> &nbsp;|&nbsp; <code>passt_base2levelmel</code> &nbsp;|&nbsp; <code>passt_base30sec</code> &nbsp;|&nbsp; <code>passt_hop100base</code> &nbsp;|&nbsp; <code>passt_hop100base2lvl</code> &nbsp;|&nbsp; <code>passt_hop100base2lvlmel</code> &nbsp;|&nbsp; <code>passt_hop160base</code> &nbsp;|&nbsp; <code>passt_hop160base2lvl</code> &nbsp;|&nbsp; <code>passt_hop160base2lvlmel</code> &nbsp;|&nbsp; <code>spectrogram</code> &nbsp;|&nbsp; <code>ssast_frame_base</code> &nbsp;|&nbsp; <code>ssast_patch_base</code> &nbsp;|&nbsp; <code>stft_mag</code> &nbsp;|&nbsp; <code>tera</code> &nbsp;|&nbsp; <code>tera_100hr</code> &nbsp;|&nbsp; <code>tera_960hr</code> &nbsp;|&nbsp; <code>tera_fbankBase_T_F_AdamW_b32_200k_100hr</code> &nbsp;|&nbsp; <code>tera_local</code> &nbsp;|&nbsp; <code>tera_logMelBase_T_F_AdamW_b32_1m_960hr</code> &nbsp;|&nbsp; <code>tera_logMelBase_T_F_AdamW_b32_1m_960hr_drop1</code> &nbsp;|&nbsp; <code>tera_logMelBase_T_F_AdamW_b32_1m_960hr_seq3k</code> &nbsp;|&nbsp; <code>tera_logMelBase_T_F_AdamW_b32_200k_100hr</code> &nbsp;|&nbsp; <code>tera_logMelBase_T_F_M_AdamW_b32_1m_960hr_drop1</code> &nbsp;|&nbsp; <code>tera_logMelBase_T_F_M_AdamW_b32_200k_100hr</code> &nbsp;|&nbsp; <code>tera_url</code> &nbsp;|&nbsp; <code>timit_posteriorgram</code> &nbsp;|&nbsp; <code>unispeech_sat</code> &nbsp;|&nbsp; <code>unispeech_sat_base</code> &nbsp;|&nbsp; <code>unispeech_sat_base_plus</code> &nbsp;|&nbsp; <code>unispeech_sat_large</code> &nbsp;|&nbsp; <code>unispeech_sat_local</code> &nbsp;|&nbsp; <code>unispeech_sat_url</code> &nbsp;|&nbsp; <code>vggish</code> &nbsp;|&nbsp; <code>vq_apc</code> &nbsp;|&nbsp; <code>vq_apc_360hr</code> &nbsp;|&nbsp; <code>vq_apc_960hr</code> &nbsp;|&nbsp; <code>vq_apc_url</code> &nbsp;|&nbsp; <code>vq_wav2vec</code> &nbsp;|&nbsp; <code>vq_wav2vec_custom</code> &nbsp;|&nbsp; <code>vq_wav2vec_gumbel</code> &nbsp;|&nbsp; <code>vq_wav2vec_kmeans</code> &nbsp;|&nbsp; <code>vq_wav2vec_kmeans_roberta</code> &nbsp;|&nbsp; <code>wav2vec</code> &nbsp;|&nbsp; <code>wav2vec2</code> &nbsp;|&nbsp; <code>wav2vec2_base_960</code> &nbsp;|&nbsp; <code>wav2vec2_base_s2st_en_librilight</code> &nbsp;|&nbsp; <code>wav2vec2_base_s2st_es_voxpopuli</code> &nbsp;|&nbsp; <code>wav2vec2_conformer_large_s2st_en_librilight</code> &nbsp;|&nbsp; <code>wav2vec2_conformer_large_s2st_es_voxpopuli</code> &nbsp;|&nbsp; <code>wav2vec2_conformer_relpos</code> &nbsp;|&nbsp; <code>wav2vec2_conformer_rope</code> &nbsp;|&nbsp; <code>wav2vec2_custom</code> &nbsp;|&nbsp; <code>wav2vec2_large_960</code> &nbsp;|&nbsp; <code>wav2vec2_large_ll60k</code> &nbsp;|&nbsp; <code>wav2vec2_large_lv60_cv_swbd_fsh</code> &nbsp;|&nbsp; <code>wav2vec2_large_voxpopuli_100k</code> &nbsp;|&nbsp; <code>wav2vec2_local</code> &nbsp;|&nbsp; <code>wav2vec2_url</code> &nbsp;|&nbsp; <code>wav2vec_custom</code> &nbsp;|&nbsp; <code>wav2vec_large</code> &nbsp;|&nbsp; <code>wav2vec_local</code> &nbsp;|&nbsp; <code>wav2vec_url</code> &nbsp;|&nbsp; <code>wavlablm_ek_40k</code> &nbsp;|&nbsp; <code>wavlablm_mk_40k</code> &nbsp;|&nbsp; <code>wavlablm_ms_40k</code> &nbsp;|&nbsp; <code>wavlm</code> &nbsp;|&nbsp; <code>wavlm_base</code> &nbsp;|&nbsp; <code>wavlm_base_plus</code> &nbsp;|&nbsp; <code>wavlm_large</code> &nbsp;|&nbsp; <code>wavlm_local</code> &nbsp;|&nbsp; <code>wavlm_url</code> &nbsp;|&nbsp; <code>xls_r_1b</code> &nbsp;|&nbsp; <code>xls_r_2b</code> &nbsp;|&nbsp; <code>xls_r_300m</code> &nbsp;|&nbsp; <code>xlsr_53</code></td>
</tr>

</table>


* **Classifier models:**

  * `aasist`
  * `sls`
  * `xlsrmamba`

* Switch SSL extractor or model via config file or command-line

* Simple commands for training and evaluation

---

## Quick Start

### 1. Configure

**YAML file (`config.yaml`):**

```yaml
ssl_feature: wavlm_large    # choose any from the list above
model_arch: aasist          # choose: aasist | sls | xlsrmamba
mode: train                 # train or eval
save_dir: output/models     # where to save models
# other settings: batch size, learning rate, etc.
```

**Or via CLI flags (model architecture is set in `config.yaml`):**

* **Example command**

```bash
python main.py --batch_size 14 --num_epochs 50 --lr 1e-6 --weight_decay 1e-4 \
  --ssl_feature <your_chosen_feature> --seed 1234 --emb_size 256 --num_encoders 12
```

### 2. Training

```bash
python main2.py --config config.yaml
```

### 3. Evaluation

```bash
python main2.py \
  --config config.yaml \
  --mode eval \
  --ckpt output/models/your_model.pth
```

---

## Switching Components

* **Change SSL feature extractor:**

  * In `config.yaml`: set `ssl_feature` to one of the names in the list above.
  * Or add `--ssl_feature <name>` on the CLI.

* **Change classifier model:**

  * In `config.yaml`: set `model_arch` to `aasist`, `sls`, or `xlsrmamba`.
  * Or add `--model_arch <name>` on the CLI.

---

## Logs & Outputs

* **Model checkpoints:** saved under the directory specified by `save_dir`.

---

## References (official repos & papers)

* S3PRL (toolkit): [https://github.com/s3prl/s3prl](https://github.com/s3prl/s3prl)
* WavLM: “WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing.”
* HuBERT: “HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units.”
* wav2vec 2.0: “A Framework for Self-Supervised Learning of Speech Representations.”
* Data2Vec (audio): “Data2Vec: A General Framework for Self-supervised Learning in Speech, Vision and Language.”
* BYOL-A: “BYOL for Audio: Self-Supervised Learning for General-Purpose Audio Representation.”
* TERA: “Self-Supervised Learning of Transformer Encoder Representation for Speech.”
* Mockingjay: “Unsupervised Speech Representation Learning with Deep Bidirectional Transformer Encoders.”
* SSAST: “Self-Supervised Audio Spectrogram Transformer.”
* PaSST: “Efficient Training of Audio Transformers with Patchout.”
* UniSpeech-SAT: “Universal Speech Representation Learning with Speaker Aware Pre-Training.”
* XLS-R / XLSR-53: Cross-lingual self-supervised models based on wav2vec 2.0.
* DeCoAR / DeCoAR 2.0: “Deep Contextualized Acoustic Representations.”
* PASE+: “Multi-task Self-Supervised Learning for Robust Speech Recognition (PASE+).”
* VQ-wav2vec: “Self-Supervised Learning of Discrete Speech Representations.”
* ContentVec: “An Improved Self-Supervised Speech Representation by Disentangling Speakers.”
* LightHuBERT / DistilHuBERT: compression/distillation variants of HuBERT.

---
