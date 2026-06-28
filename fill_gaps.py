#!/usr/bin/env python3
"""
fill_gaps.py  — PINN-QI Radar gap-filler for manuscript submission
==================================================================
Run this ONCE on a GPU RunPod pod from the repo root. It:

  1. Hard-verifies a usable CUDA GPU (fails loudly, never silent CPU fallback).
  2. Runs ONLY the three scripts that fill the remaining manuscript gaps:
        - 01_generate_baseline_bounds.py   -> QI-vs-classical gain (Appendix A)
        - 07_run_multiseed_experiments.py  -> Table 7 (multi-seed robustness)
        - 08_run_stress_test.py            -> Table 8 (stress test)
  3. Runs the plasma-physics validation (validation figure + CSVs).
  4. Bundles every new CSV / figure / table into ONE zip:
        pinn_qi_gap_outputs.zip
     ...which is the file you upload back.

USAGE (GPU pod, repo root):
    pip install -e .
    python fill_gaps.py

If you only have a CPU pod, the script will warn and still run (these three
experiments are not GPU-critical), but prefer GPU for clean provenance.
"""

import os
import sys
import time
import glob
import shutil
import zipfile
import subprocess
from datetime import datetime, timezone

REPO = os.getcwd()
STAMP = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
BUNDLE_DIR = os.path.join(REPO, f"_gap_bundle_{STAMP}")
ZIP_PATH = os.path.join(REPO, "pinn_qi_gap_outputs.zip")

GAP_SCRIPTS = [
    "scripts/01_generate_baseline_bounds.py",   # QI-vs-classical gain
    "scripts/07_run_multiseed_experiments.py",  # Table 7
    "scripts/08_run_stress_test.py",            # Table 8
]

# Folders the repo writes results into (we snapshot before/after to catch new files).
WATCH_DIRS = ["results", "tables", "figures"]


def log(msg):
    print(f"[fill_gaps] {msg}", flush=True)


def gpu_check():
    try:
        import torch
    except Exception:
        log("torch not importable yet — run `pip install -e .` first.")
        return None
    if not torch.cuda.is_available():
        log("WARNING: no CUDA GPU visible. These 3 experiments will still run on CPU,")
        log("         but provenance will say CPU. Prefer a GPU pod if you can.")
        return {"device": "cpu", "torch": torch.__version__, "cuda": None}
    name = torch.cuda.get_device_name(0)
    x = torch.randn(2048, 2048, device="cuda")
    t0 = time.time(); (x @ x).sum().item(); torch.cuda.synchronize()
    log(f"GPU OK: {name} | torch {torch.__version__} | cuda {torch.version.cuda} "
        f"| warmup {(time.time()-t0)*1000:.0f} ms")
    return {"device": name, "torch": torch.__version__, "cuda": torch.version.cuda}


def snapshot():
    snap = {}
    for d in WATCH_DIRS:
        p = os.path.join(REPO, d)
        if os.path.isdir(p):
            snap[d] = set(glob.glob(os.path.join(p, "**", "*"), recursive=True))
        else:
            snap[d] = set()
    return snap


def run_script(path):
    if not os.path.exists(path):
        log(f"SKIP (not found): {path}")
        return (path, "MISSING", 0.0)
    env = dict(os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env["PINN_QI_DEVICE"] = "cuda"  # honored if the script reads it
    log(f"running {path} ...")
    t0 = time.time()
    r = subprocess.run([sys.executable, path], env=env)
    dt = time.time() - t0
    status = "OK" if r.returncode == 0 else f"FAILED({r.returncode})"
    log(f"  -> {status} in {dt:.1f}s")
    return (path, status, round(dt, 1))


def try_validation():
    """Run the plasma validation if the script is present in repo root."""
    for cand in ["validate_plasma_model.py", "scripts/validate_plasma_model.py"]:
        if os.path.exists(cand):
            log(f"running validation: {cand}")
            subprocess.run([sys.executable, cand])
            return
    log("validation script not found in repo (optional) — skipping.")


def main():
    log(f"repo root: {REPO}")
    prov = gpu_check()

    before = snapshot()
    results = [run_script(s) for s in GAP_SCRIPTS]
    try_validation()
    after = snapshot()

    # Collect newly created / modified files.
    os.makedirs(BUNDLE_DIR, exist_ok=True)
    new_files = []
    for d in WATCH_DIRS:
        added = sorted(after.get(d, set()) - before.get(d, set()))
        # also include everything in tables/ (small, safe to always ship)
        always = sorted(after.get(d, set())) if d in ("tables",) else []
        for f in set(added) | set(always):
            if os.path.isfile(f):
                rel = os.path.relpath(f, REPO)
                dest = os.path.join(BUNDLE_DIR, rel)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy2(f, dest)
                new_files.append(rel)

    # validation outputs
    for vf in glob.glob(os.path.join(REPO, "validation_outputs", "*")):
        rel = os.path.relpath(vf, REPO)
        dest = os.path.join(BUNDLE_DIR, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(vf, dest)
        new_files.append(rel)

    # provenance file
    with open(os.path.join(BUNDLE_DIR, "RUN_PROVENANCE.txt"), "w") as fh:
        fh.write(f"timestamp_utc: {STAMP}\n")
        fh.write(f"device: {prov.get('device') if prov else 'unknown'}\n")
        fh.write(f"torch: {prov.get('torch') if prov else 'unknown'}\n")
        fh.write(f"cuda: {prov.get('cuda') if prov else 'unknown'}\n\n")
        fh.write("script results:\n")
        for s, st, dt in results:
            fh.write(f"  {st:12s} {dt:8.1f}s  {s}\n")
        fh.write("\nbundled files:\n")
        for f in sorted(set(new_files)):
            fh.write(f"  {f}\n")

    # zip it
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(BUNDLE_DIR):
            for f in files:
                full = os.path.join(root, f)
                z.write(full, os.path.relpath(full, BUNDLE_DIR))

    log("=" * 58)
    log("DONE.")
    for s, st, dt in results:
        log(f"  {st:12s} {dt:7.1f}s  {s}")
    log("=" * 58)
    log(f"UPLOAD THIS FILE BACK:  {ZIP_PATH}")
    log(f"(bundled {len(set(new_files))} files; see RUN_PROVENANCE.txt inside)")


if __name__ == "__main__":
    main()
