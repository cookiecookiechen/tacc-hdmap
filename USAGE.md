# Source-only distribution

This directory contains the simulation source, preprocessing scripts and fixed experiment configurations. It contains no trajectory samples, simulation output records, figures or statistical tables.

## Install

Python 3.10 is the reference environment.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python experiments/run_related_work_v3.py --help
```

## Inputs

Trajectory inputs must be prepared separately before running experiments. The required filenames and fingerprints are registered in `configs/related_work_v3_2.yaml`. The preprocessing entrypoints are `scripts/prepare_porto_independent.py` and `scripts/prepare_highd_small.py`; use `--help` for their arguments. The release input checker is `scripts/prepare_release_inputs.py`.

The frozen protocol expects the original selected-input fingerprints, including metadata. A different input or regenerated metadata with a different fingerprint requires its own protocol identity; do not disable checksum validation.

Original trajectories are available from the [Porto dataset provider](https://doi.org/10.24432/C55W25) and the [highD dataset provider](https://www.highd-dataset.com/), under their respective access terms.

Three files remain outside the source/configuration directories because the runner references these exact paths:

- `data/raw/porto_rsus_phase4_large.json`: RSU layout.
- `results/processed/full_sweep_plan.json`: the 486-cell parameter grid.
- `results/reference/protocol.json`: the original protocol and input fingerprints.

These are configuration/reference metadata, not raw trajectory or simulation records. The main entrypoint is `experiments/run_related_work_v3.py`; supporting experiment modules are retained as runtime and source-identity dependencies.
