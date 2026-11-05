# Running TrustSynth on a Linux research server

Use your UTHealth account and storage authorized for credentialed eICU data.
GitHub supplies code only. It does not supply patient records, synthetic records,
fitted models, private logs, or the laptop's current checkpoints. A fresh clone
starts a new run. Laptop and server jobs have independent lifecycles.

## Clone and set up

```bash
git clone https://github.com/Najmussakib93/trustsynth.git
cd trustsynth
umask 077
```

Load your server's **Python 3.10** module or environment. The package setup needs
outbound access to PyPI, PyTorch wheels and GitHub. Use an allocated compute node
according to local policy; the CUDA setup checks need an allocated NVIDIA GPU.

```bash
# NVIDIA GPU route; use --device cpu for a CPU-only run.
python3.10 server/setup.py --device cuda
```

The setup creates `.venv-sdv`, `.venv-synthcity`, `.venv-anonymeter` and `.venv-dp`.
It runs invented-data independent generator repeats, metric tests and Anonymeter
checks before marking the installation ready. A mandatory check failure stops
setup; its log is under `results/private/revision/logs/`. Existing environments
are never replaced: use a fresh clone if an incomplete setup needs rebuilding.
AIM dependency setup is bounded and may fail without blocking the other models.

SDV uses Torch 2.8.0 (CUDA 12.6 wheels); DDPM uses Torch 2.2.2/torchvision 0.17.2
(CUDA 12.1 wheels). The installed NVIDIA driver must support these wheels. Ask
your server administrator which wheel index matches its driver. The SDV index
can be changed with `--sdv-torch-index https://download.pytorch.org/whl/cu128`.
Select `--device cpu` when GPU access is unavailable. PyTorch wheel choices are
documented at https://pytorch.org/get-started/previous-versions/.

The pins in `server/requirements/` are derived from the actual working Windows
imports and their installed dependencies. Windows-only packages are omitted;
Matplotlib changes from 3.4.2 to 3.7.5 and its plotting dependencies use Python
3.10 wheels; Linux-only
IPython/XGBoost dependencies are added. Packages are installed as explicit
overlays with `--no-deps`: Synthcity's unrelated optional model requirements are
deliberately excluded using the documented lazy import patch. DDPM algorithm and
plugin sources remain unchanged. These pins are a migration configuration,
not a claim that Linux/GPU results equal the laptop's results. The setup checks
and full seed-0 independent repeats enforce reproducibility on the new host.

## Provide the credentialed data privately

Transfer your authorized source files using UTHealth's approved private transfer
method, or download them on the server using your own PhysioNet access:

```text
data/patient.csv
data/apachePatientResult.csv
data/hospital.csv
data/vitalPeriodic.csv
data/lab.csv
```

Each file may instead be supplied as `.csv.gz`. **Do not decompress large source
archives just to run this pipeline.** The reader streams compressed files and
supports a mixture of plain CSV and gzip. When both forms exist, it prefers gzip
so a partial CSV from a failed manual decompression is not used. An unreadable
gzip archive fails rather than silently falling back to the plain file.
This changes storage encoding only; cohort rules and model settings are unchanged.

Check both filesystem space and your account quota before a run. Compressed
sources save space, but generated cohorts, aggregate caches, synthetic sets,
private evaluation caches and results still require writable storage. If home
storage is insufficient, use the private project/scratch location supplied by
your UTHealth administrator, subject to its retention policy. Keep source
archives until their integrity and the storage move have been verified.

These files are ignored by Git. Do not upload them to GitHub. Optional
`vitalAperiodic.csv` and `nurseCharting.csv` were absent from the approved laptop
run. Keep that same source set for a comparable server run; adding sources is a
separate cohort change. The extractor builds the 20,000-patient cohort, preserves
missingness, checks the repeated extraction hash and uses train-only encoders.

## Submit the run

For a server that uses **Slurm**, submit from the repository root after setup:

```bash
mkdir -p results/private
sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION server/trustsynth.slurm
```

Replace the account and partition with your actual allocation; omit the account
flag if your site does not require it. The template requests one GPU, four CPU
cores, 32 GB RAM and a 48-hour limit. These are starting resource requests, not a
measured UTHealth runtime. Adjust them to the server's limits. For Slurm CPU-only
runs, remove the GPU request and change `cuda` to `cpu` in the template. If your
server uses another scheduler, use its submission instructions to run:

```bash
bash server/run.sh cuda   # or cpu, matching the setup device
```

Run lengthy jobs on compute nodes under your scheduler. The wrapper performs
repeated extraction, SDV timing/reproducibility pilots, a DDPM pilot, then the
approved ten-seed pipeline through the structural-constraint proposal. It keeps
all subprocess output in private logs. Read aggregate progress in
`results/revision/RUN_STATUS.json`; the Windows `MAIN_RUN_STATUS.json` observer
is not used on the server. Scheduler progress is available through `squeue`.

Resubmit the same command after a wall-time interruption. Validated complete
cells are reused; an interrupted fit restarts. The run stops at the Task 8(c)
structural-constraint approval gate. Final TrustScores and publication figures
remain pending that approval. Do not represent partial outputs as final results.

Proposal wording can be refreshed from the existing aggregate real-rule baseline
without loading records or rerunning training:

```bash
.venv-sdv/bin/python -m revision.constraint_proposal --proposal-only
```

The revised proposal reports population deviations and confidence intervals,
without treating an interval containing zero as evidence of preservation. These
population checks were separately approved by the author on 2026-11-04. They
remain separate from hard-rule CSR; the three real-violated rules stay in the
diagnostic tables but are excluded from primary CSR.

## Finish the completed server run after approval

For the existing completed run in `/data/nsakib1/trustsynth`, run from the
repository root:

```bash
cd /data/nsakib1/trustsynth
git pull --ff-only origin master
umask 077
mkdir -p results/private
nohup .venv-sdv/bin/python -m revision.finalize --run > results/private/finalization-launcher.log 2>&1 &
echo "Finalization PID: $!"
```

This uses CPU evaluation and existing private caches. It does not run setup,
extract the cohort again or retrain any generator; no GPU selection is needed.
It validates the forty main cells, six mechanism cells, independent repeat checks
and prerequisite metrics before continuing. Missing prerequisites cause a safe
stop rather than an automatic training rerun. Concurrent finalizers are blocked.
The initial short bootstrap pilot writes a provisional runtime estimate to
`results/revision/finalization_runtime_estimate.json`.

Monitor progress without opening patient files:

```bash
cat results/private/finalization-launcher.log
cat results/revision/RUN_STATUS.json
cat results/revision/finalization_runtime_estimate.json
```

When `FINALIZATION_STATUS.json` says `complete`, read
`results/revision/REVISION_REPORT.md`. Download
`results/revision/trustsynth_aggregate_review.zip` for aggregate review. Its
explicit allowlist excludes records, models, predictions, distances and private
logs. Publication TIFF originals remain under `results/revision/figures/`;
the review archive includes PNG figures. Do not upload `data/` or
`results/private/`. If interrupted, repeat the same finalization command;
validated integrity cells are reused. After completion,
`.venv-sdv/bin/python 03_generate_outputs.py` regenerates aggregate reports and
figures without loading patient records.

GitHub does not carry the laptop's ignored caches or live status. Do not combine
CPU and GPU seed results or assume bit-for-bit equality across operating systems,
package versions or hardware. Keep the new host's receipts and environment
snapshots with its results.
