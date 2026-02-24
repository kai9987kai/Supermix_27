
# Supermix_27

Supermix_27 is a **packaged project release** that separates:

- **Open-source build/training source code**
- **A trained Python runtime model package**
- **A static web UI bundle for GitHub Pages**
- **Datasets / runtime databases / prebuilt distribution bundles**

This repository is structured for **GitHub publishing with Git + Git LFS**, so large artifacts (models, datasets, databases, zip bundles) can be tracked correctly.

---

## Overview

This repo contains both:

1. **The real trained model runtime (Python/PyTorch)** for local execution
2. **A static browser UI** for GitHub Pages hosting

These are **not the same runtime**:

- The Python runtime uses the actual `.pth` model file
- The GitHub Pages site is static and cannot execute Python/PyTorch directly in-browser

That means the Pages version is intended as a **static/retrieval-style interface**, while the Python runtime is where the **real model inference** runs.

---

## Repository Layout

```text
Supermix_27/
├─ source/            # Open-source code for dataset building, training, manifests, helpers, web source files
├─ runtime_python/    # Trained model (.pth), metadata, Python terminal interface, Python web interface
├─ web_static/        # Static GitHub Pages chat UI bundle (browser-side metadata retrieval)
├─ datasets/          # Retained .jsonl datasets for the pipeline
├─ databases/         # Runtime/chat SQLite databases and sidecar files
├─ bundles/           # Prebuilt zip bundles + upload manifest + upload guide
├─ .gitattributes     # Git LFS tracking rules
├─ LICENSE            # MIT License
└─ README.md
````

---

## What’s Included

### `source/`

Open-source project code used to build datasets and train the model stack.

Typical contents (as described in the current repo notes):

* training scripts
* dataset builders
* manifests
* helper modules
* static web source assets

### `runtime_python/`

Runtime package for local Python execution, including:

* trained model (`.pth`)
* metadata
* Python terminal interface
* Python web interface

### `web_static/`

Static front-end bundle intended for **GitHub Pages** hosting.

This version uses browser-side metadata retrieval and does **not** run the `.pth` model directly in the browser.

### `datasets/`

Current retained `.jsonl` datasets used by this pipeline.

### `databases/`

Runtime/chat SQLite database files and sidecars (e.g. WAL/SHM).

### `bundles/`

Prebuilt zip bundles and publishing helpers, including:

* source bundle
* runtime bundle
* datasets bundle
* databases bundle
* GitHub Pages/static bundle
* upload manifest / upload guide

---

## Important Runtime Distinction (Python model vs Static Web)

This project includes a real trained model file in the Python runtime package, and the Python web interface uses that model directly.

However, **GitHub Pages is static hosting**, so it cannot run a local Python/PyTorch `.pth` model.

### Practical implication

* ✅ `runtime_python/` = real model execution (local Python environment)
* ✅ `web_static/` = static UI / metadata-driven browser experience
* ❌ GitHub Pages cannot run the `.pth` model directly

### If you want true in-browser model inference

You would need to:

* convert the model to a browser-compatible runtime format (for example ONNX)
* implement browser inference (e.g. WebGPU/WebAssembly stack)
* port the runtime pipeline from Python to JavaScript/TypeScript

---

## Quick Start

## 1) Clone the repository (with Git LFS support)

```bash
git clone https://github.com/kai9987kai/Supermix_27.git
cd Supermix_27
git lfs install
git lfs pull
```

> `git lfs pull` is important to download the real large files (models, datasets, DBs, zip bundles) instead of just pointers.

---

## 2) Run the Python runtime (real model path)

This repo’s notes indicate the Python runtime contains a trained `.pth` model and a Python web app entrypoint.

Example (if you want the local Python web interface):

```bash
python runtime_python/chat_web_app.py
```

If the project also includes a terminal/CLI entry script, run it from `runtime_python/` (script name may vary depending on your local package contents).

### Recommended setup (example)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

# Install dependencies using the repo's dependency files (if included)
# e.g. requirements.txt / pyproject.toml inside runtime_python/ or source/
```

> If dependency files are split between `source/` and `runtime_python/`, install both sets as needed for training vs inference workflows.

---

## 3) Use the static web UI (GitHub Pages / local static preview)

The `web_static/` folder is designed for static hosting.

### Local preview options

* Open the main HTML file directly in a browser (if self-contained)
* Or serve the folder with a simple local server:

```bash
# Python 3
cd web_static
python -m http.server 8000
```

Then open:

```text
http://localhost:8000
```

### GitHub Pages deployment

Publish the contents of `web_static/` to:

* GitHub Pages branch/folder (`gh-pages` or `/docs`)
* or another static host (Netlify / Vercel static / Cloudflare Pages)

---

## Git LFS Configuration

This repository is configured to track large files with Git LFS, including patterns like:

* `*.pth`
* `*.jsonl`
* `*.db`, `*.db-wal`, `*.db-shm`
* `bundles/*.zip`

### Verify LFS is working

```bash
git lfs ls-files
```

### Common issue

If files are tiny pointer files instead of real assets:

```bash
git lfs install
git lfs pull
```

---

## Publishing / Pushing to GitHub

```bash
git lfs install
git add .
git commit -m "Add Supermix_27 source, runtime, datasets, DBs, and web bundles"
git push -u origin main
```

---

## Bundles & Manifests

The `bundles/` directory includes release-oriented packaging assets and helper docs.

Notable files (mentioned in repo notes):

* `bundles/champion_github_upload_manifest.json` — bundle inventory + checksums
* `bundles/GITHUB_UPLOAD_GUIDE.txt` — guidance on what is repo-safe vs better stored in LFS/Releases

### Suggested use

* Keep frequently used project assets in the repo (LFS where appropriate)
* Move very large release payloads to **GitHub Releases** when practical
* Use the manifest file to verify bundle integrity and contents

---

## Development Workflow Suggestions

### Training / data pipeline work

Use `source/` when you want to:

* build datasets
* retrain / fine-tune models
* regenerate manifests
* modify helper modules or training scripts

### Runtime / inference work

Use `runtime_python/` when you want to:

* run the trained model locally
* test Python terminal chat/inference
* run the Python web UI

### Static showcase / Pages work

Use `web_static/` when you want to:

* host a browser UI on GitHub Pages
* prototype front-end UX without Python backend execution

---

## Data & Database Notes

Because this repo contains `datasets/` and `databases/`:

* be careful committing sensitive or private records
* review SQLite contents before publishing
* consider sanitized/demo data for public releases
* document schema and retention policy if the repo is shared widely

---

## Troubleshooting

### GitHub Pages site loads but doesn’t behave like local Python runtime

This is expected if the browser UI is static-only. GitHub Pages cannot run the `.pth` model.

### Large files missing after clone

Run:

```bash
git lfs install
git lfs pull
```

### Python runtime fails to start

* Verify Python version compatibility
* Install required dependencies from repo files
* Confirm the `.pth` model and metadata are present in `runtime_python/`
* Check paths expected by the runtime scripts

### Database-related errors

* Ensure SQLite DB files and sidecars were pulled via LFS
* Confirm file permissions
* Avoid mixing stale `.db-wal` / `.db-shm` files across environments

---

## License

MIT License (see `LICENSE`).

---

## Credits / Maintainer

**Kai Piper** (`kai9987kai`)

If you use or extend this repo, consider documenting:

* model version lineage
* dataset provenance
* runtime expectations (Python version, dependencies, GPU/CPU support)
* static vs dynamic deployment behavior

---

## Roadmap Ideas (Optional)

* [ ] Add explicit `requirements.txt` / `pyproject.toml` for `runtime_python/`
* [ ] Add a one-command launcher script (`run_local.bat` / `run_local.sh`)
* [ ] Add model metadata schema documentation
* [ ] Add database schema docs
* [ ] Add ONNX export path for browser-capable inference experiments
* [ ] Add GitHub Pages deployment workflow (`.github/workflows/pages.yml`)


[1]: https://github.com/kai9987kai/Supermix_27 "GitHub - kai9987kai/Supermix_27"
