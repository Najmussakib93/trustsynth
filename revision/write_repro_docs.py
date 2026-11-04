"""Generate current commands and preserve the previous README as historical text."""
import json
import subprocess
from .safe_reporting import ROOT, PUBLIC


def main():
    readme = ROOT / 'README.md'
    old = ROOT / 'revision' / 'legacy' / 'README_before_revision.md'
    if not old.exists():
        old.write_text(readme.read_text(encoding='utf-8'), encoding='utf-8')
    inventory_path = PUBLIC / 'audit' / 'inventory.json'
    remote = (json.loads(inventory_path.read_text())['git_remote'] if inventory_path.exists()
        else subprocess.check_output(['git', 'remote', 'get-url', 'origin'], cwd=ROOT, text=True).strip())
    text = f'''# TrustSynth: eICU revision

TrustSynth evaluates synthetic structured healthcare data across fidelity,
privacy, fairness, predictive utility and structural integrity. This revision
uses an approved, missingness-preserving 20,000-patient eICU-CRD v2.0 cohort.
The full revision is in progress; see `results/revision/REVISION_REPORT.md`
and `results/revision/MAIN_RUN_STATUS.json` for completed work and pending experiments.
The main status observer is separate from optional-pilot status messages.

## Linux / UTHealth server

Follow [server/README.md](server/README.md) for Python 3.10, isolated CPU/CUDA
setup, smoke checks, private eICU provisioning and a Slurm submission template.
Start with `python3.10 server/setup.py --device cuda` on an allocated GPU node,
then submit `server/trustsynth.slurm` with your actual account and partition.
GitHub carries code, not the ignored data, checkpoints or live experiment outputs.
A fresh clone starts a new run; Windows outputs do not prove GPU reproducibility.

## Access and confidentiality

Obtain authorized [PhysioNet eICU-CRD v2.0 access](https://physionet.org/content/eicu-crd/2.0/)
and comply with its credentialing, training and data-use agreement. Place the
authorized CSV files in `data/`: `patient.csv`, `apachePatientResult.csv`,
`hospital.csv`, `vitalPeriodic.csv`, and `lab.csv`. Each source may also remain
compressed as `.csv.gz`; mixed plain/gzip inputs are supported. Gzip takes
priority if both encodings exist. Optional additional sources
are `vitalAperiodic.csv` and `nurseCharting.csv`; both were absent in this run.
Never publish real or derived synthetic records, fitted models, identifiers,
predictions, nearest-record distances or raw debugging logs. Patient-level
artifacts stay in ignored `data/` and `results/private/`; public outputs are
aggregate tables with small counts suppressed below 11.

## Protocol

The executable configuration is `revision/protocol.py`; its JSON snapshot is
`results/revision/protocol.json`. Seeds 0–9 each redraw a stratified 80/20 mortality
split, fit on train only and sample its size. CTGAN and TVAE use 300 epochs;
mechanism checks use seeds 0–2 with CTGAN log-frequency disabled or TVAE at
900 epochs. The cohort is retrospective: selected intact stays, all-stay valid
observation means, discharge LOS, observed mortality and APACHE IV. Age above
89 is encoded as 90. There is no admission-time prediction claim.

## Commands (PowerShell, from the repository root)

```powershell
python -m unittest revision.test_audit revision.test_preprocessing revision.test_metrics
python 01_extract_cohort.py
python 01_extract_cohort.py  # validate repeat hash, reuse source-keyed aggregates
python -m revision.benchmark --device cpu
python 02_run_evaluation.py --stage through-constraints --device cpu
python 03_generate_outputs.py
```

The supervisor temporarily inhibits automatic Windows system sleep while active;
display settings and permanent power settings are not changed. Its steps can be
resumed with the same command: validated completed artifacts are reused. A failed
worker stops execution and records its phase; raw third-party error logs remain
private. A ten-seed SDV run with full seed-0 repeats and mechanism checks was
estimated at about five active hours from measured epoch times; DDPM and privacy
evaluation add time, and host interruptions can extend elapsed runtime.

Individual stages are `gaussian`, `train`, `quality`, `privacy`, `anonymeter`, and
`through-constraints`. Individual cells can be rerun with:

```powershell
python -m revision.generators --generator CTGAN --seed 0 --epochs 300 --repeat
python -m revision.evaluate --generator CTGAN --seed 0 --phase quality
.venv-anonymeter/Scripts/python.exe -m revision.evaluate --generator CTGAN --seed 0 --phase privacy
```

## Environments

The observed host uses Python 3.9.5. SDV/model evaluation currently uses the
working CPU installation; the attempted CUDA installation did not pass timely
initialization and is not the experiment route. Isolated `.venv-synthcity`,
`.venv-dp` and `.venv-anonymeter` environments hold optional-library differences.
Exact imported distributions are emitted by executed workers to
`results/revision/environments/requirements-*-pinned.txt`. These are execution
snapshots, not a claim that an untested environment is reproducible.

To reconstruct one of those tested worker environments, create a clean venv and
install its complete emitted lock with `pip install --no-deps -r <lockfile>`.
The CPU wheel index is included in the lock. Smoke-check it using
`python -m revision.inspect_environment <kind>`. For the base CPU SDV environment,
use the generator pilot instead of the CUDA-specific `sdv` smoke-check.

Synthcity 0.2.12 requires a matching older CPU Torch/torchvision and Opacus. The
isolated setup uses Torch 2.2.2+cpu, torchvision 0.17.2+cpu and Opacus 1.4.0.
`python -m revision.patch_synthcity` copies its installed source into the isolated
environment and defers unused metrics/time-series imports; the patch and SHA
verification of unchanged DDPM algorithm/plugin files are in the revision
outputs. Its scope must be disclosed with the package version. It does not
replace the DDPM implementation.

AIM setup is bounded by `python -m revision.setup_dp`, using SmartNoise and its
recommended private-pgm commit. AIM uses fixed public domains and bins, including
a missing bin, rather than non-private train medians or learned category support.
Its epsilon labels identify nominal mechanism budgets. Publicly seeded research
runs and non-private evaluation outputs are not a certified end-to-end DP release;
the abstract mechanism's guarantee assumes protected random coins and a fixed
training table. The attempt status is written to
`dp_setup.json` and `dp_attempt.json` under revision outputs.

Anonymeter uses version 1.0.0 and Numba 0.58.1 in its separate environment. Exact
attacker auxiliary columns and secrets are declared before execution in
`revision/attacker_config.json`. Its unseeded NumPy generator factory and the
module-level singling-out generator are scoped to deterministic seeded streams;
repeated attacks are checked before accepting seed-0 results.

## Outputs and remaining gate

The combined metric table is `results/revision/per_seed_results.csv`, with
`seed,generator,metric,value`. Per-cell subgroup, representation, parity and
privacy aggregates are under `results/revision/cells/`. Generator receipts are
under `results/revision/generators/`; patient-level caches are ignored.

The original audit and manuscript-number mapping remain under
`results/revision/audit/`. Legacy source is frozen in `revision/legacy/` and the
audit entry points use that schema. The original cohort/results are preserved.
New structural tests stop at the Task 8(c) author-approval gate. Final scores,
paired statistics, supplementary methods and 500-dpi final figures follow that
approval; the current interim report must not be represented as a finished paper.

Configured Git remote: `{remote}`. This differs from the manuscript URL
`github.com/Najmus-Sakib/TrustSynth`; remote ownership has not been inferred.
The manuscript is read-only for this task. Publishing code does not publish
credentialed records or mark the revised experiments complete.
'''
    readme.write_text(text, encoding='utf-8')
    (ROOT / 'revision' / 'README.md').write_text('# Revision execution\n\nSee the root README for the current protocol, exact commands, environments, aggregate outputs and approval gate.\n', encoding='utf-8')
    print('Reproducibility commands updated; old README archived.', flush=True)


if __name__ == '__main__':
    main()
