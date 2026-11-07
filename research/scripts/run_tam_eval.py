#!/usr/bin/env python3
"""
GraphSMOTE + TAM evaluation runner (CPU).
TAM = Song et al. ICML 2022; GraphSMOTE = Zhao et al. WSDM 2021.
Append-only CSV; does not delete STA/baseline artifacts.
Does not re-run BlogCatalog seed-42 baselines (already exist).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".venv" / "bin" / "python"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

LOG_DIR = ROOT / "research" / "logs" / "tam"
CFG_DIR = ROOT / "research" / "configs"
RES_DIR = ROOT / "research" / "results"
CKPT_DIR = ROOT / "research" / "checkpoints"
for d in (LOG_DIR, CFG_DIR, RES_DIR, CKPT_DIR):
    d.mkdir(parents=True, exist_ok=True)

TAM_CSV = RES_DIR / "tam_results.csv"

CSV_FIELDS = [
    "run_id", "dataset", "setting", "seed", "epochs", "imbalance", "im_ratio",
    "device", "model", "tam", "tam_alpha", "tam_beta", "tam_phi", "tam_delta",
    "tam_warmup", "tam_scope",
    "command", "start_utc", "end_utc", "duration_sec", "exit_code",
    "final_test_loss", "final_test_accuracy", "final_test_auc_roc",
    "final_test_macro_f1", "final_test_bal_acc", "final_test_minority_f1",
    "final_test_minority_recall", "final_test_pr_auc",
    "last_valid_auc_roc", "last_valid_macro_f1",
    "checkpoint_path", "log_path", "config_path", "env_fingerprint",
    "has_traceback", "has_nan", "notes",
]


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def env_fingerprint():
    code = (
        "import torch,numpy,scipy,sklearn,networkx,platform;"
        "print(f'py={platform.python_version()};torch={torch.__version__};"
        "cuda={torch.cuda.is_available()};numpy={numpy.__version__}')"
    )
    try:
        return subprocess.check_output([str(PYTHON), "-c", code], cwd=ROOT, text=True).strip()
    except Exception as e:
        return f"err:{e}"


def parse_metrics(log_text: str):
    out = {k: None for k in [
        "final_test_loss", "final_test_accuracy", "final_test_auc_roc",
        "final_test_macro_f1", "final_test_bal_acc", "final_test_minority_f1",
        "final_test_minority_recall", "final_test_pr_auc",
        "last_valid_auc_roc", "last_valid_macro_f1",
    ]}
    test_loss_acc = list(re.finditer(
        r"Test set results:\s*loss=\s*([0-9.]+)\s*accuracy=\s*([0-9.]+)", log_text))
    if test_loss_acc:
        m = test_loss_acc[-1]
        out["final_test_loss"] = float(m.group(1))
        out["final_test_accuracy"] = float(m.group(2))
    test_auc = list(re.finditer(
        r"test current auc-roc score:\s*([0-9.]+),\s*current macro_F score:\s*([0-9.]+)", log_text))
    if test_auc:
        m = test_auc[-1]
        out["final_test_auc_roc"] = float(m.group(1))
        out["final_test_macro_f1"] = float(m.group(2))
    test_ext = list(re.finditer(
        r"test extended metrics:\s*bal_acc=([nan0-9.]+),\s*minority_f1=([nan0-9.]+),\s*"
        r"minority_recall=([nan0-9.]+),\s*pr_auc=([nan0-9.]+)", log_text))
    if test_ext:
        m = test_ext[-1]
        def _f(x):
            return float("nan") if x == "nan" else float(x)
        out["final_test_bal_acc"] = _f(m.group(1))
        out["final_test_minority_f1"] = _f(m.group(2))
        out["final_test_minority_recall"] = _f(m.group(3))
        out["final_test_pr_auc"] = _f(m.group(4))
    val_auc = list(re.finditer(
        r"valid current auc-roc score:\s*([0-9.]+),\s*current macro_F score:\s*([0-9.]+)", log_text))
    if val_auc:
        m = val_auc[-1]
        out["last_valid_auc_roc"] = float(m.group(1))
        out["last_valid_macro_f1"] = float(m.group(2))
    return out


def ensure_csv():
    if not TAM_CSV.exists():
        with TAM_CSV.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


def append_row(row: dict):
    ensure_csv()
    with TAM_CSV.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore").writerow(
            {k: row.get(k, "") for k in CSV_FIELDS}
        )


def already_done(run_id: str) -> bool:
    if not TAM_CSV.exists():
        return False
    with TAM_CSV.open() as f:
        for row in csv.DictReader(f):
            if row.get("run_id") == run_id and str(row.get("exit_code")) == "0":
                return True
    return False


def prepare_pretrained(dataset: str, seed: int) -> bool:
    """Install *seed-isolated* Pretrained into checkpoint/{dataset}/.

    Never fall back to a leftover Pretrained from another seed (that contaminated
    BlogCatalog seed0 when seed42 weights remained in checkpoint/BlogCatalog/).
    """
    src_dir = CKPT_DIR / f"seed{seed}_{dataset}"
    dst_dir = ROOT / "checkpoint" / dataset
    dst_dir.mkdir(parents=True, exist_ok=True)
    pre = src_dir / "Pretrained.pth"
    if pre.exists():
        shutil.copy2(pre, dst_dir / "Pretrained.pth")
        return True
    cands = sorted(src_dir.glob("recon_*_False_0.5.pth"))
    if cands:
        shutil.copy2(cands[-1], dst_dir / "Pretrained.pth")
        shutil.copy2(cands[-1], src_dir / "Pretrained.pth")
        return True
    # Do NOT return True for unrelated leftover checkpoint/{dataset}/Pretrained.pth
    return False


def archive_checkpoints(dataset: str, seed: int, setting: str, tam: bool):
    src_dir = ROOT / "checkpoint" / dataset
    dst_dir = CKPT_DIR / f"seed{seed}_{dataset}"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for p in src_dir.glob(f"{setting}_*.pth"):
        tag = "tam" if tam else "gs"
        shutil.copy2(p, dst_dir / f"{tag}_{p.name}")
    if setting == "recon":
        lasts = sorted(src_dir.glob("recon_*_False_0.5.pth"))
        if lasts:
            shutil.copy2(lasts[-1], dst_dir / "Pretrained.pth")
            shutil.copy2(lasts[-1], src_dir / "Pretrained.pth")


def estimate_eta(dataset: str, setting: str, epochs: int, n_runs: int) -> str:
    # Rough CPU ETAs from prior STA work on this VM
    per_epoch = {
        ("cora", "no"): 0.02,
        ("cora", "newG_cls"): 0.06,
        ("cora", "recon_newG"): 0.08,
        ("cora", "recon"): 0.09,
        ("BlogCatalog", "no"): 0.35,
        ("BlogCatalog", "newG_cls"): 1.8,
        ("BlogCatalog", "recon_newG"): 2.5,
        ("BlogCatalog", "recon"): 2.5,
    }
    key = (dataset, setting if setting != "recon" else "recon")
    # map tam settings to base
    base = setting.replace("+tam", "")
    sec = per_epoch.get((dataset, base), per_epoch.get((dataset, "newG_cls"), 0.1)) * epochs
    total = sec * n_runs
    return f"~{sec/60:.1f} min/run × {n_runs} ≈ {total/3600:.2f} h"


def run_one(dataset, setting, seed, epochs, tam=False, extra=None, notes="", force=False):
    extra = extra or []
    tam_tag = "tam1" if tam else "tam0"
    run_id = f"tam_{dataset}_{setting}_seed{seed}_e{epochs}_{tam_tag}"
    if not force and already_done(run_id):
        print(f"[SKIP] {run_id}")
        return None

    cfg_path = CFG_DIR / f"{run_id}.json"
    log_path = LOG_DIR / f"{run_id}.log"
    cmd = [
        str(PYTHON), "main.py",
        "--imbalance", "--no-cuda",
        f"--dataset={dataset}",
        f"--setting={setting}",
        f"--seed={seed}",
        f"--epochs={epochs}",
        f"--config_out={cfg_path}",
    ]
    if tam:
        cmd.append("--tam")
    cmd += list(extra)

    needs_load = setting in ("newG_cls", "recon_newG", "sta_newG_cls", "sta_recon_newG")
    if needs_load:
        prepare_pretrained(dataset, seed)
        cmd += ["--load=Pretrained"]

    print(f"[ETA] {estimate_eta(dataset, setting, epochs, 1)}")
    print(f"[RUN] {' '.join(cmd)}")
    start = utc_now()
    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w") as lf:
        lf.write(f"START {start}\nCMD {' '.join(cmd)}\n\n")
        lf.flush()
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=lf, stderr=subprocess.STDOUT, env=env)
        end = utc_now()
        dur = int(round(time.time() - t0))
        lf.write(f"\nEND {end} exit={proc.returncode} duration_sec={dur}\n")

    text = log_path.read_text(errors="replace")
    metrics = parse_metrics(text)
    archive_checkpoints(dataset, seed, setting, tam)

    row = {
        "run_id": run_id,
        "dataset": dataset,
        "setting": setting,
        "seed": seed,
        "epochs": epochs,
        "imbalance": True,
        "im_ratio": 0.5,
        "device": "cpu",
        "model": "sage",
        "tam": tam,
        "tam_alpha": 2.5,
        "tam_beta": 0.5,
        "tam_phi": 1.2,
        "tam_delta": 0.4,
        "tam_warmup": 5,
        "tam_scope": "train_orig",
        "command": " ".join(cmd),
        "start_utc": start,
        "end_utc": end,
        "duration_sec": dur,
        "exit_code": proc.returncode,
        **metrics,
        "checkpoint_path": str(CKPT_DIR / f"seed{seed}_{dataset}"),
        "log_path": str(log_path.relative_to(ROOT)),
        "config_path": str(cfg_path.relative_to(ROOT)) if cfg_path.exists() else "",
        "env_fingerprint": env_fingerprint(),
        "has_traceback": "Traceback" in text,
        "has_nan": bool(re.search(r"\bNaN\b", text)),
        "notes": notes or "GraphSMOTE+TAM hybrid; Song et al. ICML 2022; NOT novelty claim",
    }
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        for k in ("tam", "tam_alpha", "tam_beta", "tam_phi", "tam_delta", "tam_warmup", "tam_scope"):
            if k in cfg:
                row[k] = cfg[k]
    append_row(row)
    print(f"[DONE] {run_id} exit={proc.returncode} auc={row.get('final_test_auc_roc')} f1={row.get('final_test_macro_f1')} ({dur}s)")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["smoke", "cora_primary", "blog_tam", "one", "eta"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=2010)
    ap.add_argument("--setting", type=str, default="newG_cls")
    ap.add_argument("--dataset", type=str, default="cora")
    ap.add_argument("--tam", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.mode == "eta":
        for ds, st, n in [
            ("cora", "newG_cls", 10),
            ("cora", "recon_newG", 10),
            ("BlogCatalog", "newG_cls", 5),
            ("BlogCatalog", "recon_newG", 5),
        ]:
            print(ds, st, estimate_eta(ds, st, args.epochs, n))
        return

    if args.mode == "smoke":
        # Need > warmup epochs so TAM actually activates (warmup default 5 ⇒ need epoch index >=5).
        smoke_e = max(args.epochs, 12)
        run_one("cora", "recon", args.seed, smoke_e, tam=False, notes="smoke recon", force=True)
        run_one("cora", "newG_cls", args.seed, smoke_e, tam=False, notes="smoke GS off", force=True)
        run_one("cora", "newG_cls", args.seed, smoke_e, tam=True, notes="smoke GS+TAM", force=True)
        # Verify TAM log lines exist
        log = LOG_DIR / f"tam_cora_newG_cls_seed{args.seed}_e{smoke_e}_tam1.log"
        text = log.read_text(errors="replace") if log.exists() else ""
        assert "[TAM] active" in text, "TAM did not activate in smoke — check warmup/epochs"
        print("SMOKE_TAM_ACTIVE_OK")
        return

    if args.mode == "one":
        run_one(args.dataset, args.setting, args.seed, args.epochs, tam=args.tam, force=args.force)
        return

    if args.mode == "cora_primary":
        seeds = [42, 0, 1, 2, 3]
        print("[ETA]", estimate_eta("cora", "newG_cls", args.epochs, len(seeds) * 2),
              "+", estimate_eta("cora", "recon_newG", args.epochs, len(seeds) * 2))
        for seed in seeds:
            # Ensure Pretrained exists (reuse STA checkpoints when present)
            if not prepare_pretrained("cora", seed):
                run_one("cora", "recon", seed, args.epochs, tam=False,
                        notes="TAM-phase recon pretrain", force=args.force)
            # Matched GS without TAM (re-run for fairness if missing in tam_results)
            run_one("cora", "newG_cls", seed, args.epochs, tam=False,
                    notes="matched GS newG_cls (TAM-off control)", force=args.force)
            run_one("cora", "newG_cls", seed, args.epochs, tam=True,
                    notes="matched GS+TAM newG_cls", force=args.force)
            run_one("cora", "recon_newG", seed, args.epochs, tam=False,
                    notes="matched GS recon_newG (TAM-off control)", force=args.force)
            run_one("cora", "recon_newG", seed, args.epochs, tam=True,
                    notes="matched GS+TAM recon_newG", force=args.force)
        return

    if args.mode == "blog_tam":
        seeds = [42, 0, 1, 2, 3]
        print("[NOTE] Skipping BlogCatalog seed-42 *baselines* (already exist from STA worker).")
        print("[NOTE] Running GraphSMOTE ± TAM finetunes only.")
        print("[ETA]", estimate_eta("BlogCatalog", "newG_cls", args.epochs, len(seeds) * 2),
              "+", estimate_eta("BlogCatalog", "recon_newG", args.epochs, len(seeds) * 2))
        for seed in seeds:
            if not prepare_pretrained("BlogCatalog", seed):
                # Need recon for Pretrained; seed42 may already have it from STA
                run_one("BlogCatalog", "recon", seed, args.epochs, tam=False,
                        notes="BlogCatalog recon for TAM (not a baseline duplicate of no/up/reweight/embed)",
                        force=args.force)
            run_one("BlogCatalog", "newG_cls", seed, args.epochs, tam=False,
                    notes="BlogCatalog GS control for TAM", force=args.force)
            run_one("BlogCatalog", "newG_cls", seed, args.epochs, tam=True,
                    notes="BlogCatalog GS+TAM", force=args.force)
            run_one("BlogCatalog", "recon_newG", seed, args.epochs, tam=False,
                    notes="BlogCatalog GS recon_newG control", force=args.force)
            run_one("BlogCatalog", "recon_newG", seed, args.epochs, tam=True,
                    notes="BlogCatalog GS+TAM recon_newG", force=args.force)
        return


if __name__ == "__main__":
    main()
