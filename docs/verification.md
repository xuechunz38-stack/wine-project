# Verification record — September 29, 2026

## Verified

- **Local Python:** macOS arm64, Python 3.12.14; fixed direct dependencies from
  `requirements-dev.txt`. Black and Ruff pass. **70 tests passed in 28.75s**.
  [Saved pytest output](evidence/pytest-local.txt).
- **Local analysis:** `python analysis.py` completed, wrote both PNGs and both
  CSVs, and reproduced the earlier model metrics. The fixed 20% tasting budget
  selected 213 test observations and found 99/127 good wines with the two models.
- **Local Docker build:** Docker Desktop 4.88.1 / Engine 29.7.2, Linux arm64
  engine. Official Python 3.11 slim base pinned by manifest digest. BuildKit
  successfully exported `wine-project:week4`.
  [Unabridged successful build log](evidence/docker-build.txt).
- **GitHub Actions:** [run 36646890746](https://github.com/xuechunz38-stack/wine-project/actions/runs/36646890746)
  for commit `1cdec4ee9fbc1081712a6ef25974d1ddd6f84554` succeeded. Both Python
  3.11/3.12 jobs passed formatting, lint, and tests. The independent Docker job
  built the image, ran the full analysis, checked output files, and uploaded
  the `wine-analysis` artifact.

<img src="../screenshots/week4_ci.png" alt="Actual successful CI run with two Python jobs and a container job" width="820">

## Local runtime check still pending

The Mac session became locked during verification. Docker image construction
completed, but both a bind-mounted run and a run using only the image's own
filesystem remained in `Created` state without analysis output. The computer
interface could not inspect the desktop while locked. The precise Docker
startup cause has not yet been diagnosed; this is **not** recorded as a
successful local container run. Local container tests and local/container CSV
comparison are also not yet verified. Unlock the Mac, inspect Docker Desktop,
and rerun the documented commands to finish this assignment requirement.

The CI container result demonstrates working container execution on the GitHub
runner; it does not substitute for the assignment's requested local run.

## Reproducibility boundaries

The base image digest and direct Python dependency versions are fixed, but
all transitive wheels are not hash-locked. Exact image bytes are therefore not
guaranteed across rebuilds. Model seeds and the committed CSV provide repeatable
analysis; benchmark timings and plot rendering can differ across machines.
GitHub's run also reports upstream Node-action deprecation notices; these did
not fail any job.
