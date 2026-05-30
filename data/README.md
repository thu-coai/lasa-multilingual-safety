# Bundled LASA Data

This directory contains the public data bundle used by the LASA analysis, SSI
training, and Latent KTO scripts.

## Files

- `safety_train_translated.json`
  - 1,000 harmful training prompts with multilingual translations.
  - Format: a JSON list with `idx`, `original_query`, and `translations`.
  - Kept at the data root for bottleneck-analysis compatibility.

- `ssi/unsafe_train.json`
  - 11,000 harmful SSI-training examples.
  - Built from the translated safety data plus HarmBench training prompts.
  - Format: a JSON list with normalized `prompt`, `language`, `lingual`,
    `source`, and `idx` fields.

- `ssi/safety_train_translated.json`
  - Source translated safety component used to build `ssi/unsafe_train.json`.

- `ssi/harmbench_training.json`
  - Source HarmBench training component used to build `ssi/unsafe_train.json`.

- `training/mixed_dpo_ultra_safety_converted.json`
  - Bundled Latent KTO training data.
  - Format: paired preference JSON with `prompt`, `chosen`, `rejected`, and
    `source`; optional metadata fields such as `language` and `category` may
    also be present.

- `ultrafeedback/monolingual_first1000_<lang>.json`
  - 1,000 benign UltraFeedback prompts per language.
  - Languages: `en`, `zh`, `ko`, `th`, `sw`, `bn`, `ar`, `it`, `jw`, `vi`.
  - Format: a JSON list with `idx`, `prompt`, `chosen`, and `rejected`.

The code uses `jw` as the Javanese filename code.
