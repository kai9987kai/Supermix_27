# Supermix_27

This repository has been prepared for GitHub publishing with a split between normal Git files and Git LFS files.

## What is included

- `source/`
  - Open-source project code needed to build datasets and train the model
  - Training scripts, builders, manifests, helper modules, and static web source files
- `runtime_python/`
  - The real trained model (`.pth`) + metadata + Python terminal interface + Python web interface
- `web_static/`
  - GitHub Pages static chat UI bundle (browser retrieval from metadata JSON)
- `datasets/`
  - Current retained `.jsonl` datasets for this pipeline
- `databases/`
  - Chat/runtime SQLite DB files and sidecars currently present
- `bundles/`
  - Prebuilt zip bundles (`source`, `runtime`, `datasets`, `databases`, GitHub Pages static bundle) plus upload manifest/guide

## Important: same model vs static web

- The **same real trained model** is in `runtime_python/champion_model_chat_supermix_v27_500k_ft.pth`.
- The Python web interface (`runtime_python/chat_web_app.py`) uses that real `.pth` model.
- GitHub Pages is static hosting and **cannot execute a Python/PyTorch `.pth` model** in-browser.
- The `web_static/` site is included for convenience, but it uses metadata retrieval in JavaScript rather than executing the `.pth` model.

If you want the actual neural model running in a browser, the model must be converted to a browser runtime format (for example ONNX/WebGPU) and the pipeline must be ported.

## Git LFS tracking (configured)

This repo is configured to track large files via Git LFS:

- `*.pth`
- `*.jsonl`
- `*.db`, `*.db-wal`, `*.db-shm`
- `bundles/*.zip`

## Publish to GitHub

1. Create or connect a GitHub remote for this repo.
2. Run `git lfs install` (already installed locally on this machine).
3. Commit and push normally:
   - `git add .`
   - `git commit -m "Add Supermix_27 source, runtime, datasets, DBs, and web bundles"`
   - `git push -u origin main`

## Notes

- `bundles/champion_github_upload_manifest.json` contains checksums and bundle inventory.
- `bundles/GITHUB_UPLOAD_GUIDE.txt` summarizes what is repo-safe and what is typically better in GitHub Releases/LFS.
