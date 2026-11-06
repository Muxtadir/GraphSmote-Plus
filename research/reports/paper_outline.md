# Paper Outline — Matched Empirical Study of GraphSMOTE + TAM

**Date:** 2026-11-06 15:22 PT
**Branch:** `cursor/evidence-wording-polish-8ee5`
**Status:** **Outline only** — scaffolding for a future draft; **not** IEEE paper body text and not a results paper.
**Evidence pack:** `research/reports/final_evidence_summary.md`

Store mirror: `/cursor/stores/self/docs/reports/paper_outline.md`.

---

## Working title (honest)

*An Empirical Evaluation of Topology-Aware Margin Loss Integrated with GraphSMOTE for Imbalanced Node Classification*

Alt: *Matched multi-seed evidence on Cora and BlogCatalog; STA attachment as a null ablation.*

Not acceptable titles: anything implying a new TAM invention, universal official defaults, or SOTA without matched competitor runs.

---

## 1. Abstract (bullets)

- Empirical **integration study**: apply Song et al. TAM inside Zhao et al. GraphSMOTE (prior art + prior art).
- Locked **Song-reference settings plus explicitly documented integration settings**; seeds {42,0,1,2,3}; epochs 2010; CPU stack.
- Finding: GS+TAM **produced higher mean** Acc/AUC/macro-F1 than matched GS, with **positive paired differences** (5/5 seeds) on Cora and BlogCatalog.
- STA soft attachment = **null** vs matched GS.
- Limits: citation-only competitors; CPU compatibility-adjusted; two datasets; primary n=5 descriptive (no significance); Blog simple baselines n=4; Blog STA seed-42 only.

---

## 2. Introduction

1. Imbalanced node classification context.
2. Synthesis-side vs loss-side lines.
3. Gap: matched evidence for TAM **on** GraphSMOTE without changing the synthesizer.
4. Narrow contributions: matched GS±TAM tables; STA null; val-only OFAT with primary HPs unchanged.
5. Non-claims: not inventing TAM; not SOTA; not “universal official defaults.”

---

## 3. Related work

Cite GraphSMOTE + TAM primarily; GraphENS/GATSMOTE/GraphSHA/GraphSANN as synthesis contemporaries (**citation-only**). Differentiation: our hybrid = keep synthesizer, add TAM loss; STA = failed attachment branch.

---

## 4. Method (integration, not invention)

- GraphSMOTE unchanged (Zhao et al.).
- TAM ACM/ADM attributed to Song et al.; wiring in `tam.py`; `tam_scope=train_orig`; detached margins.
- Primary HPs: **Song-reference settings** (α=2.5, β=0.5, φ=1.2, warmup=5) **plus explicitly documented integration settings** (δ=0.4 / `--tam_delta`, `tam_scope=train_orig`). Treat δ cautiously as a port integration knob (wired to `temp_gamma` in `tam.py`), not as a Song-paper “official default” claim.
- STA: null ablation only (Cora matched; BlogCatalog STA seed-42 only).
- Metrics: define via `utils.py` `print_class_acc`. Note **descriptive** Cora Acc≈bal_acc vs BlogCatalog Acc≠bal_acc without inventing unaudited support counts. Runtime = **descriptive** CPU wall-clock `duration_sec` (secondary).

---

## 5. Experiments

- **Causal tables:** matched GS±TAM from `tam_results.csv` only.
- **Context tables:** older non-matched baselines (`baseline_results.csv`, BlogCatalog simple) — labeled non-matched.
- STA matched Cora null table.
- OFAT: label α=4.0 / warmup=0 as **validation-only project sensitivity candidates**; primary unchanged.
- No BlogCatalog TAM sweep; no competitor ports.
- Cautious wording: “produced higher mean metrics”, “positive paired differences” — not “beats/improves” without paired context.

Suggested tables/figs: T1 Cora matched; T2 Blog matched; T3 context baselines; T4 STA null; T5 OFAT candidates; F1 paired ΔF1; F2 synthesis vs loss map.

---

## 6. Results narrative

1. Cora matched ΔAUC ≈ +0.0077 / +0.0054; ΔF1 ≈ +0.020; 5/5 ΔF1>0; Acc≈bal_acc (descriptive).
2. Blog matched ΔAUC ≈ +0.0104 / +0.0127; ΔF1 ≈ +0.0073 / +0.0056; 5/5 ΔF1>0; Acc≠bal_acc (descriptive).
3. Context baselines lower than GS; not for TAM effect sizes.
4. STA null.
5. Sensitivity candidates appendix-only.

---

## 7–9. Discussion / Limitations / Conclusion

- Loss-level hybrid showed positive paired differences; attachment STA did not.
- Copy evidence §8 limitations: citation-only competitors; BlogCatalog simple baselines **n=4**; BlogCatalog STA **seed-42 only**; primary n=5 **descriptive** (no significance); CPU runtime descriptive; matched vs non-matched discipline.
- Conclusion: empirical GS+TAM **integration study** with descriptive positive paired differences under Song-reference + documented integration settings; STA null; not a new TAM; not SOTA.
- This file remains an **outline**, not paper body.

---

## 10. Reproducibility checklist

- [ ] Env fingerprint; commands; configs; CSVs
- [ ] Metric defs cite `utils.py`; TAM wiring cites `tam.py` + Song et al.
- [ ] Primary = Song-reference + documented integration settings
- [ ] Sensitivity candidates labeled validation-only
- [ ] Matched vs non-matched tables separated
- [ ] Runtime described as CPU wall-clock secondary

---

## 11. Forbidden language

- “We propose TAM …” / novel topology-aware margin invented here
- “Universal official TAM defaults …” / treating δ as Song-paper official default without integration caveat
- “Beats / improves / outperforms …” without matched paired context
- SOTA vs GraphENS/GraphSHA/… without matched runs (competitors are citation-only)
- “STA significantly improves GraphSMOTE” / multi-seed BlogCatalog STA
- Mixing non-matched absolute GS with TAM effect sizes
- Promoting OFAT candidates as primary
- Significance claims from n=5

Acceptable: “We evaluate Song et al.’s TAM integrated into Zhao et al.’s GraphSMOTE under a matched protocol.”
