# Research-Guided Upgrades Applied

This repo now includes practical upgrades inspired by recent preference-optimization research for chat alignment.

## Papers Used

1. `DPO` (Rafailov et al., 2023):  
   https://arxiv.org/abs/2305.18290
2. `ORPO` (Hong et al., 2024):  
   https://aclanthology.org/2024.emnlp-main.626/
3. `SimPO` (Meng et al., 2024):  
   https://arxiv.org/abs/2405.14734
4. `RE-PO` (Cao et al., 2025):  
   https://arxiv.org/abs/2509.24159
5. `RPO: Reward-aware Preference Optimization: A Unified Mathematical Framework for Model Alignment` (Sun et al., 2025):  
   https://arxiv.org/abs/2502.00203
6. `LMPO: Length-Controlled Margin-Based Preference Optimization without Reference Model` (Li et al., 2025):  
   https://arxiv.org/abs/2502.14643

## What Was Implemented

1. Pairwise preference objective during fine-tuning:
- Added to `finetune_chat.py` as `_preference_loss(...)`.
- Enabled with `--pref_weight` and `--pref_beta`.
- Combines cross-entropy with a preference-style margin objective (chosen class vs sampled negative class).
- Added hard-negative mining via `--hard_negative_ratio` to preferentially train against confusable classes.
- Added objective selection via `--pref_objective`:
  - `sigmoid` (SimPO/ORPO-style pairwise logistic)
  - `repo_relu` (RePO-style max-margin ReLU objective)

2. Robust preference weighting for noisy data:
- Added `--adaptive_pref_weighting` in `finetune_chat.py`.
- Confidence-weighted preference terms approximate robust/noisy-preference handling from recent work.

3. Expectation-style grouped preference estimation:
- Added `--pref_group_size` for multi-negative preference estimation.
- Added `--pref_group_estimator epo` for expectation-style group reduction over sampled negatives.
- This provides a practical grouped-estimation upgrade for noisy/sparse preference settings and aligns with recent RPO-style design guidance on multi-response preference estimation.

4. Class-imbalance mitigation:
- Added `--balanced_sampler` in `finetune_chat.py` using inverse-frequency sampling.

5. Better retrieval-time response selection:
- Updated `chat_app.py` to fuse `--top_labels` predicted buckets instead of a single bucket.
- Updated `chat_pipeline.py` scoring to include query-context similarity, bucket confidence, and diversity penalties.
- Added response sampling control with `--response_temperature`.
- Added stronger response cleanup in `chat_pipeline.py` to remove near-duplicate clauses and filler fragments.

6. Capacity and style upgrades:
- Added an `xlarge` model option in `model_variants.py` with a dual-adapter routed classifier head.
- Added an `xxlarge` model option in `model_variants.py` with tri-branch routed adapters for higher-capacity classification.
- Extended `finetune_chat.py` and `chat_app.py` with `--model_size xlarge` and `--extra_expansion_dim` support.
- Extended `finetune_chat.py` and `chat_app.py` with `--model_size xxlarge` and `--third_expansion_dim` support.
- Added style-aware reranking in `chat_pipeline.py` plus automatic style inference (`balanced`, `creative`, `concise`, `analyst`).
- Added `--style_mode` and `--creativity` controls in `chat_app.py` for more creative, controllable responses.

7. Reliability upgrades for larger runs:
- Added `--grad_accum_steps` in `finetune_chat.py` for stable large-model optimization with bigger effective batch size.
- Added EMA shadow weights via `--ema_decay` with EMA-based evaluation/saving enabled by default (can disable with `--disable_ema_eval`).

8. Dataset scale upgrade:
- Added `build_super_dataset.py` to merge multiple JSONL corpora, apply quality filtering, dedupe, and scale to a larger training set.

## Usage Example

```bash
python finetune_chat.py --data conversation_data.hybrid_v5_clean.jsonl --weights champion_model_chat_large_ft_v5.pth --model_size large --train_all --balanced_sampler --split_mode stratified --pref_weight 0.22 --pref_beta 2.6 --pref_objective sigmoid --pref_group_size 4 --pref_group_estimator epo --hard_negative_ratio 0.78 --adaptive_pref_weighting --pref_warmup_epochs 1.2 --lr_schedule cosine --warmup_steps 180 --early_stop_patience 2 --epochs 2 --batch_size 128 --device cpu --output champion_model_chat_large_ft_v6.pth --meta chat_model_large_meta_v6.json
```

```bash
python chat_app.py --weights champion_model_chat_large_ft_v6.pth --meta chat_model_large_meta_v6.json --device cpu --pool_mode all --top_labels 4 --response_temperature 0.08 --llm_db llm_chat_v5.db --db_top_k 160
```
