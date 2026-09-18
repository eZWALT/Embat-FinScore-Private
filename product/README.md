<div align="center">

# product/

**Runnable FinScore application** (scaffold)

<p>
  <img alt="Python" src="https://img.shields.io/badge/python-3.12-3776AB.svg">
  <img alt="uv" src="https://img.shields.io/badge/uv-pyproject.toml-de5fe9.svg">
  <img alt="Docker" src="https://img.shields.io/badge/docker-supported-2496ED.svg">
</p>

</div>

Python package for **Aura** (HackSpain 2026 X Ray / Embat), private working tree. Dependencies live in `pyproject.toml` and are installed with [uv](https://docs.astral.sh/uv/).

The brief requires a **trajectory score** (not only a last-month still), an **explanation**, and a **sellable layer on top**. The buyer is not chosen yet.

This package does **not** compute a health score yet. `python -m embat_finscore` only prints name and version so the install and image can be checked.

## Table of Contents

- [Requirements](#requirements)
- [Install](#install)
- [Usage](#usage)
- [Docker](#docker)
- [Layout](#layout)

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- Docker, only for the container path

## Install

From this directory:

```bash
uv sync
```

`uv.lock` pins the resolved set. Edit `pyproject.toml` when you add a real dependency, then run `uv lock` and `uv sync` again.

## Usage

```bash
uv run python -m embat_finscore
```

Expected output today:

```text
Embat FinScore
version 0.1.0
HackSpain 2026 — Embat Track
Scaffold only. No scoring model is implemented yet.
```

## Docker

Build and run from this directory:

```bash
docker build -t embat-finscore .
docker run --rm embat-finscore
```

The image copies `uv` from `ghcr.io/astral-sh/uv` and installs the project with `uv sync --frozen --no-dev`, following [uv's Docker guide](https://docs.astral.sh/uv/guides/integration/docker/).

## Layout

```text
product/
├── Dockerfile
├── .dockerignore
├── .python-version
├── pyproject.toml
├── uv.lock
├── README.md
└── src/embat_finscore/
    ├── __init__.py
    └── __main__.py
```

Application code goes under `src/embat_finscore/`. Keep datasets in `../data/` and notebooks in `../analysis/`.
