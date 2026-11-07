#!/usr/bin/env python3
"""Aggregate tam_results.csv; paired Δ; figures under research/figures/tam/."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "research" / "results"
FIG = ROOT / "research" / "figures" / "tam"
FIG.mkdir(parents=True, exist_ok=True)
CSV_PATH = RES / "tam_results.csv"


def ms(vals):
    vals = [float(v) for v in vals if v not in ("", None)]
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan"), 0
    m = sum(vals) / n
    s = 0.0 if n == 1 else math.sqrt(sum((x - m) ** 2 for x in vals) / (n - 1))
    return m, s, n


def main():
    rows = [r for r in csv.DictReader(CSV_PATH.open()) if "e2010" in r["run_id"]]
    lines = ["# TAM aggregate (recomputed from tam_results.csv)", "",
             "Attribution: GraphSMOTE=Zhao et al. WSDM 2021; TAM=Song et al. ICML 2022 (hybrid).",
             ""]
    lines.append("| Dataset | Setting | TAM | n | Acc mean±std | AUC mean±std | macro-F1 mean±std |")
    lines.append("|---------|---------|-----|--:|-------------:|-------------:|------------------:|")

    series = {}
    for dataset in sorted(set(r["dataset"] for r in rows)):
        for setting in ["newG_cls", "recon_newG"]:
            for tam in [False, True]:
                rr = [r for r in rows if r["dataset"] == dataset and r["setting"] == setting
                      and str(r["tam"]).lower() == str(tam).lower()]
                if not rr:
                    continue
                acc = ms([r["final_test_accuracy"] for r in rr])
                auc = ms([r["final_test_auc_roc"] for r in rr])
                f1 = ms([r["final_test_macro_f1"] for r in rr])
                lines.append(
                    f"| {dataset} | `{setting}` | {tam} | {auc[2]} | "
                    f"{acc[0]:.4f}±{acc[1]:.4f} | {auc[0]:.4f}±{auc[1]:.4f} | {f1[0]:.4f}±{f1[1]:.4f} |"
                )
                series[(dataset, setting, tam)] = (auc, f1, rr)

    lines.append("")
    lines.append("## Paired Δ = (GS+TAM) − GS (Cora)")
    lines.append("")
    lines.append("| Setting | Seed | Δ AUC | Δ macro-F1 |")
    lines.append("|---------|-----:|------:|-----------:|")
    for setting in ["newG_cls", "recon_newG"]:
        das, dfs = [], []
        for seed in ["42", "0", "1", "2", "3"]:
            a = [r for r in rows if r["dataset"] == "cora" and r["setting"] == setting
                 and r["seed"] == seed and str(r["tam"]).lower() == "false"]
            b = [r for r in rows if r["dataset"] == "cora" and r["setting"] == setting
                 and r["seed"] == seed and str(r["tam"]).lower() == "true"]
            if not a or not b:
                continue
            da = float(b[0]["final_test_auc_roc"]) - float(a[0]["final_test_auc_roc"])
            df = float(b[0]["final_test_macro_f1"]) - float(a[0]["final_test_macro_f1"])
            das.append(da); dfs.append(df)
            lines.append(f"| `{setting}` | {seed} | {da:+.4f} | {df:+.4f} |")
        if das:
            lines.append(f"| `{setting}` | mean | {sum(das)/len(das):+.4f} | {sum(dfs)/len(dfs):+.4f} |")

    out = RES / "tam_aggregate.md"
    out.write_text("\n".join(lines) + "\n")
    print(out.read_text())

    # Cora bar figure
    labels, aucs, auc_e, f1s, f1_e = [], [], [], [], []
    for setting, tam in [("newG_cls", False), ("newG_cls", True), ("recon_newG", False), ("recon_newG", True)]:
        key = ("cora", setting, tam)
        if key not in series:
            continue
        auc, f1, _ = series[key]
        labels.append(f"{setting}\nTAM={tam}")
        aucs.append(auc[0]); auc_e.append(auc[1])
        f1s.append(f1[0]); f1_e.append(f1[1])
    if labels:
        x = np.arange(len(labels))
        fig, ax = plt.subplots(1, 2, figsize=(9, 4))
        ax[0].bar(x, aucs, yerr=auc_e, capsize=3, color=["#2f6f8f", "#3a9b7a"] * 2)
        ax[0].set_xticks(x); ax[0].set_xticklabels(labels, fontsize=8)
        ax[0].set_ylabel("Test AUC-ROC"); ax[0].set_title("Cora matched GS ± TAM")
        ax[1].bar(x, f1s, yerr=f1_e, capsize=3, color=["#b85c38", "#c47b2c"] * 2)
        ax[1].set_xticks(x); ax[1].set_xticklabels(labels, fontsize=8)
        ax[1].set_ylabel("Test macro-F1"); ax[1].set_title("Cora matched GS ± TAM")
        fig.tight_layout()
        fig.savefig(FIG / "cora_gs_vs_tam_auc_f1.png", dpi=150)
        print("wrote", FIG / "cora_gs_vs_tam_auc_f1.png")

    # Paired delta figure
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.35
    seeds = ["42", "0", "1", "2", "3"]
    for i, setting in enumerate(["newG_cls", "recon_newG"]):
        deltas = []
        for seed in seeds:
            a = [r for r in rows if r["dataset"] == "cora" and r["setting"] == setting
                 and r["seed"] == seed and str(r["tam"]).lower() == "false"]
            b = [r for r in rows if r["dataset"] == "cora" and r["setting"] == setting
                 and r["seed"] == seed and str(r["tam"]).lower() == "true"]
            deltas.append(float(b[0]["final_test_macro_f1"]) - float(a[0]["final_test_macro_f1"]) if a and b else 0)
        xpos = np.arange(len(seeds)) + (i - 0.5) * width
        ax.bar(xpos, deltas, width=width, label=setting)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(np.arange(len(seeds))); ax.set_xticklabels(seeds)
    ax.set_xlabel("Seed"); ax.set_ylabel("Δ macro-F1 (TAM − GS)")
    ax.set_title("Cora paired F1 gains from TAM")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "cora_tam_paired_delta_f1.png", dpi=150)
    print("wrote", FIG / "cora_tam_paired_delta_f1.png")


if __name__ == "__main__":
    main()
