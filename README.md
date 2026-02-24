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
