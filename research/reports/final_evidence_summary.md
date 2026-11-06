# Final Evidence Summary — GraphSMOTE + TAM Hybrid

**Date:** 2026-11-06 15:22 PT
**Branch:** `cursor/evidence-wording-polish-8ee5`
**Document role:** Draft-readiness evidence pack (documentation only).
**Framing:** Matched **empirical integration study** of GraphSMOTE + TAM (Zhao et al. WSDM 2021 + Song et al. ICML 2022). We do **not** invent TAM; we do **not** claim SOTA. STA is a documented **null** edge-attachment ablation.
**Stack:** compatibility-adjusted PyTorch 2.5.1+cpu — **not** original-paper table claims.
**Rule for this document:** No new experiments; no competitor re-runs; no CSV/code/logs/datasets/config edits; numbers preserved from existing CSVs/reports.

Store mirror: `/cursor/stores/self/docs/reports/final_evidence_summary.md`.

---

## 0. Verdict

| Question | Answer |
|----------|--------|
| Matched GS+TAM vs matched GS (primary HPs)? | On this stack, GS+TAM **produced higher mean** Acc/AUC/macro-F1 (and related metrics) than matched GS for Cora and BlogCatalog × `{newG_cls, recon_newG}`, with **positive paired differences** ΔAUC/ΔF1 on **5/5 seeds** in every primary cell. |
| Did we invent TAM? | **No** — Song et al. ICML 2022. Primary uses **Song-reference settings plus explicitly documented integration settings** (§2). |
| Is this a SOTA claim? | **No** — competitors not re-run under our fingerprint; citation-only. |
| STA vs matched GraphSMOTE? | **Null** — matched Cora STA ≈ GS within error bars; not a positive method claim. |
| Primary HPs after Cora OFAT? | **Unchanged** — α=4.0 and warmup=0 are **validation-only project sensitivity candidates**, not applied to primary tables. |

Primary matrix: **40/40** exit-0 numeric cells (`tam_final_audit.md` + recheck §12). Missing-controls P0–P3: **PASS**.

**Drafting stance:** Ready for *cautious* IEEE drafting as an **empirical GS+TAM integration study**, not a new TAM method paper and not a SOTA paper.

---

## 1. Citations (primary)

- **GraphSMOTE:** Zhao, T., Zhang, X., & Wang, S. (2021). *GraphSMOTE: Imbalanced Node Classification on Graphs with Graph Neural Networks.* WSDM ’21. https://doi.org/10.1145/3437963.3441720
- **TAM:** Song, J., Park, J., & Yang, E. (2022). *TAM: Topology-Aware Margin Loss for Class-Imbalanced Node Classification.* ICML 2022. https://proceedings.mlr.press/v162/song22a.html — code https://github.com/Jaeyun-Song/TAM
- **Related (cite only; not re-run here):** GraphENS (Park et al., ICLR 2022); GATSMOTE (Liu et al., Mathematics 2022); GraphSHA (Li et al., KDD 2023); GraphSANN (Liu et al., IJCAI 2023). See `tam_novelty_review.md` / `competing_methods_feasibility.md`.

---

## 2. Locked protocol (primary)

| Knob | Value |
|------|-------|
| Datasets | Cora (artificial imbalance), BlogCatalog |
| Settings | `newG_cls`, `recon_newG` (±TAM) |
| Seeds | `{42,0,1,2,3}` |
| Epochs / model / device | `2010` / GraphSAGE / CPU `--no-cuda` |
| Imbalance | `--imbalance --im_ratio=0.5` |
| Pretrain load | per-seed `recon` → `--load=Pretrained` |
| TAM (when on) | **Song-reference settings** α=2.5, β=0.5, φ=1.2, warmup=5 **plus explicitly documented integration settings** δ=0.4 (`--tam_delta`), `tam_scope=train_orig` |
| Integration wiring | `tam.py` ACM/ADM on CE logits; margins `@torch.no_grad()`; unlabeled nodes contribute temperature-scaled softmax preds (not gold labels); scope `train_orig` |
| Metrics | Acc, ROC-AUC, macro-F1, bal_acc, minority F1/recall, PR-AUC, runtime_s (§2.1–2.3) |
| Std | sample stdev (n−1) |

**HP naming (mandatory):** Do **not** call the primary bundle “universal official TAM defaults.” Split clearly:
- **Song-reference settings (this project):** α=2.5, β=0.5, φ=1.2, warmup=5 — chosen to align with common Song et al. / upstream TAM argument names and reference values used in our port.
- **Explicitly documented integration settings (this GraphSMOTE port):** δ=0.4 via `--tam_delta` (passed into `tam.py` as the temperature-style margin knob named `temp_gamma` in that module), `tam_scope=train_orig`, detached margins, CE-path wiring.
- **δ wording (cautious):** Treat δ/`--tam_delta` as an **integration hyperparameter of this port**, documented in code and configs. Do **not** assert that δ is a named Song-paper “official default,” or that our CLI name is identical to every upstream release’s public API naming—only that our implementation wires this value into the ACM/ADM temperature-style term as coded in `tam.py`.
Primary tables stay locked to this Song-reference + documented-integration bundle.

Source CSV: `research/results/tam_results.csv`. Aggregate: `research/results/tam_aggregate.md`.

### 2.1 Metric definitions (code: `utils.py` → `print_class_acc`)

Evaluated on the held-out split’s logits (`output`) and labels; predictions `y_pred = argmax(logits)`, probabilities `probs = softmax(logits)`.

| Metric | Definition in this repo | Code |
|--------|-------------------------|------|
| **Accuracy** | Mean of `y_pred == y_true` over evaluated nodes (also via `accuracy()`). | `utils.py` `accuracy` / `print_class_acc` |
| **ROC-AUC** | `sklearn.metrics.roc_auc_score`: multiclass → `average='macro'`, `multi_class='ovr'` on `probs`; binary → `probs[:,1]`. | `utils.py` ~L311–314 |
| **macro-F1** | `sklearn.metrics.f1_score(y_true, y_pred, average='macro')`. | `utils.py` ~L316 |
| **Balanced accuracy** | `sklearn.metrics.balanced_accuracy_score(y_true, y_pred)` (= mean per-class recall). | `utils.py` ~L334 |
| **Minority F1** | Macro-F1 on minority IDs = last `im_class_num` classes (GraphSMOTE convention), `f1_score(..., labels=minority_classes, average='macro', zero_division=0)`. | `utils.py` ~L320–342 |
| **Minority recall** | Same minority label set; `recall_score(..., average='macro', zero_division=0)`. | `utils.py` ~L343–345 |
| **PR-AUC** | Mean OvR average precision: for each class `c` present in `y_true`, `average_precision_score((y_true==c), probs[:,c])`, then `mean(aps)`. | `utils.py` ~L349–355 |

CSV harvest parses printed `auc-roc` / `macro_F` and `extended metrics: bal_acc=…, minority_f1=…, minority_recall=…, pr_auc=…` (e.g. `research/scripts/run_tam_eval.py`).

### 2.2 Acc vs balanced accuracy on Cora and BlogCatalog (descriptive)

On **Cora** primary rows in §3, Acc and bal_acc **agree to the reported precision** (e.g. both `0.7283±0.0126` for GS `newG_cls`). Balanced accuracy is mean per-class recall (`sklearn.metrics.balanced_accuracy_score`). Numerically equal Acc and bal_acc is **consistent with** roughly equal per-class evaluation supports, but this pass does **not** invent or assert audited class-count tables; report the observed equality as a descriptive fact from the metrics only.

On **BlogCatalog** primary rows, Acc and bal_acc **diverge numerically** (e.g. GS `newG_cls` Acc `0.2522` vs bal_acc `0.1138`). Report both. Do not treat Cora Acc≈bal_acc as a general identity, as proof of “easy” data, or as a claim about unaudited support counts.

### 2.3 Runtime semantics

| Field | Meaning |
|-------|---------|
| `runtime_s` / `duration_sec` | Wall-clock seconds for that **single** `main.py` invocation on this CPU VM (from runner start/end timestamps). |
| Not | GPU paper runtime; amortized multi-job cost; network I/O-normalized time; or a claim of algorithmic complexity. |
| Caveats | CPU-only (`torch 2.5.1+cpu`); can inflate under host contention; BlogCatalog ≫ Cora; `recon_newG` includes heavier decode/finetune path variability (large std). |
| Use in paper | **Descriptive** secondary/appendix resource note only; **not** a primary quality metric for method ranking. |

---

## 3. Primary results — matched GS±TAM (causal comparison)

**Causal comparison only.** Same dataset, setting, seed, epochs, Pretrained lineage; only `--tam` differs. Source: `tam_results.csv` / `tam_aggregate.md`. Do **not** mix with §4 absolute levels.

### 3.1 Mean ± std (n=5)

| Dataset | Setting | TAM | Acc | AUC | macro-F1 | bal_acc | min_F1 | min_recall | PR-AUC | runtime_s |
|---------|---------|-----|----:|----:|---------:|--------:|-------:|-----------:|-------:|----------:|
| cora | `newG_cls` | False | 0.7283±0.0126 | 0.9395±0.0065 | 0.7251±0.0132 | 0.7283±0.0125 | 0.7108±0.0203 | 0.6485±0.0306 | 0.8095±0.0147 | 171.0±15.6 |
| cora | `newG_cls` | True | 0.7470±0.0228 | 0.9472±0.0054 | 0.7449±0.0241 | 0.7470±0.0228 | 0.7296±0.0481 | 0.6739±0.0652 | 0.8270±0.0164 | 191.8±25.7 |
| cora | `recon_newG` | False | 0.7314±0.0263 | 0.9410±0.0114 | 0.7281±0.0245 | 0.7314±0.0263 | 0.7191±0.0294 | 0.6545±0.0395 | 0.8128±0.0275 | 334.8±243.7 |
| cora | `recon_newG` | True | 0.7501±0.0220 | 0.9464±0.0124 | 0.7481±0.0207 | 0.7501±0.0220 | 0.7363±0.0302 | 0.6909±0.0462 | 0.8257±0.0287 | 367.8±236.6 |
| BlogCatalog | `newG_cls` | False | 0.2522±0.0388 | 0.6504±0.0290 | 0.1130±0.0256 | 0.1138±0.0235 | 0.0368±0.0147 | 0.0283±0.0107 | 0.0980±0.0214 | 3827.2±60.1 |
| BlogCatalog | `newG_cls` | True | 0.2577±0.0359 | 0.6608±0.0308 | 0.1203±0.0251 | 0.1207±0.0228 | 0.0437±0.0171 | 0.0329±0.0132 | 0.1053±0.0235 | 4682.4±723.2 |
| BlogCatalog | `recon_newG` | False | 0.2626±0.0181 | 0.6577±0.0255 | 0.1272±0.0208 | 0.1269±0.0171 | 0.0494±0.0208 | 0.0428±0.0173 | 0.1059±0.0152 | 6755.8±2270.3 |
| BlogCatalog | `recon_newG` | True | 0.2718±0.0174 | 0.6704±0.0230 | 0.1328±0.0197 | 0.1341±0.0172 | 0.0505±0.0231 | 0.0475±0.0191 | 0.1123±0.0151 | 6356.6±2032.8 |

### 3.2 Paired Δ = (GS+TAM) − GS (mean over seeds)

| Dataset | Setting | mean ΔAUC | mean ΔF1 | mean ΔAcc | mean Δbal_acc | mean Δmin_F1 | mean ΔPR-AUC | #seeds ΔF1>0 |
|---------|---------|----------:|---------:|----------:|--------------:|-------------:|-------------:|-------------:|
| cora | `newG_cls` | +0.0077 | +0.0198 | +0.0187 | +0.0187 | +0.0188 | +0.0175 | 5/5 |
| cora | `recon_newG` | +0.0054 | +0.0199 | +0.0187 | +0.0187 | +0.0172 | +0.0129 | 5/5 |
| BlogCatalog | `newG_cls` | +0.0104 | +0.0073 | +0.0055 | +0.0069 | +0.0069 | +0.0073 | 5/5 |
| BlogCatalog | `recon_newG` | +0.0127 | +0.0056 | +0.0092 | +0.0071 | +0.0011 | +0.0063 | 5/5 |

**Stats caution:** Primary summaries are **descriptive** mean±std over n=5 seeds plus paired signs. Do **not** claim statistical significance, p-values, or formal hypothesis tests from n=5 alone.

### 3.3 Per-seed primary metrics + paired Δ

#### cora

##### `newG_cls`

| Seed | GS Acc | GS AUC | GS F1 | TAM Acc | TAM AUC | TAM F1 | ΔAUC | ΔF1 |
|-----:|-------:|-------:|------:|--------:|--------:|-------:|-----:|-----:|
| 42 | 0.7403 | 0.9413 | 0.7390 | 0.7455 | 0.9437 | 0.7440 | +0.0024 | +0.0050 |
| 0 | 0.7169 | 0.9346 | 0.7099 | 0.7481 | 0.9449 | 0.7448 | +0.0103 | +0.0349 |
| 1 | 0.7429 | 0.9475 | 0.7389 | 0.7844 | 0.9546 | 0.7845 | +0.0071 | +0.0456 |
| 2 | 0.7169 | 0.9313 | 0.7177 | 0.7273 | 0.9417 | 0.7273 | +0.0104 | +0.0096 |
| 3 | 0.7247 | 0.9427 | 0.7199 | 0.7299 | 0.9509 | 0.7240 | +0.0082 | +0.0041 |

##### `recon_newG`

| Seed | GS Acc | GS AUC | GS F1 | TAM Acc | TAM AUC | TAM F1 | ΔAUC | ΔF1 |
|-----:|-------:|-------:|------:|--------:|--------:|-------:|-----:|-----:|
| 42 | 0.6883 | 0.9223 | 0.6906 | 0.7117 | 0.9247 | 0.7128 | +0.0024 | +0.0222 |
| 0 | 0.7247 | 0.9425 | 0.7188 | 0.7532 | 0.9503 | 0.7496 | +0.0077 | +0.0309 |
| 1 | 0.7506 | 0.9521 | 0.7494 | 0.7662 | 0.9554 | 0.7664 | +0.0033 | +0.0170 |
| 2 | 0.7506 | 0.9403 | 0.7486 | 0.7584 | 0.9493 | 0.7575 | +0.0089 | +0.0089 |
| 3 | 0.7429 | 0.9476 | 0.7333 | 0.7610 | 0.9525 | 0.7541 | +0.0049 | +0.0208 |

#### BlogCatalog

##### `newG_cls`

| Seed | GS Acc | GS AUC | GS F1 | TAM Acc | TAM AUC | TAM F1 | ΔAUC | ΔF1 |
|-----:|-------:|-------:|------:|--------:|--------:|-------:|-----:|-----:|
| 42 | 0.1980 | 0.6139 | 0.0777 | 0.2037 | 0.6215 | 0.0865 | +0.0076 | +0.0088 |
| 0 | 0.2281 | 0.6325 | 0.1000 | 0.2443 | 0.6474 | 0.1097 | +0.0149 | +0.0098 |
| 1 | 0.2612 | 0.6544 | 0.1123 | 0.2614 | 0.6608 | 0.1162 | +0.0064 | +0.0040 |
| 2 | 0.2843 | 0.6609 | 0.1360 | 0.2862 | 0.6685 | 0.1387 | +0.0075 | +0.0027 |
| 3 | 0.2895 | 0.6902 | 0.1392 | 0.2930 | 0.7057 | 0.1504 | +0.0155 | +0.0112 |

##### `recon_newG`

| Seed | GS Acc | GS AUC | GS F1 | TAM Acc | TAM AUC | TAM F1 | ΔAUC | ΔF1 |
|-----:|-------:|-------:|------:|--------:|--------:|-------:|-----:|-----:|
| 42 | 0.2377 | 0.6276 | 0.1026 | 0.2503 | 0.6465 | 0.1110 | +0.0189 | +0.0084 |
| 0 | 0.2528 | 0.6404 | 0.1132 | 0.2602 | 0.6545 | 0.1191 | +0.0141 | +0.0059 |
| 1 | 0.2629 | 0.6602 | 0.1241 | 0.2705 | 0.6708 | 0.1271 | +0.0106 | +0.0030 |
| 2 | 0.2789 | 0.6664 | 0.1442 | 0.2851 | 0.6741 | 0.1507 | +0.0077 | +0.0065 |
| 3 | 0.2808 | 0.6938 | 0.1522 | 0.2928 | 0.7061 | 0.1560 | +0.0123 | +0.0039 |

Full metric columns (bal_acc, min_F1/recall, PR-AUC, runtime) live in `research/results/tam_aggregate.md`.

---

## 4. Older non-matched baseline reproductions (context only)

**Not causal for TAM.** Absolute GraphSMOTE levels here can differ from §3 matched GS-off rows due to different Pretrained lineage / runner generations. Use only as **context controls**, never for TAM effect sizes.

### 4.1 Cora simple / GraphSMOTE reproduction CSV (n=5)

From `research/results/baseline_results.csv` / `baseline_reproduction.md` (compatibility-adjusted; **not** paper tables).

| Setting | Acc | AUC | macro-F1 |
|---------|----:|----:|---------:|
| `no` | 0.6441±0.0191 | 0.9065±0.0091 | 0.6348±0.0190 |
| `upsampling` | 0.6493±0.0299 | 0.9084±0.0097 | 0.6414±0.0301 |
| `reweight` | 0.6483±0.0280 | 0.9078±0.0093 | 0.6410±0.0291 |
| `embed_up` | 0.6499±0.0234 | 0.9074±0.0096 | 0.6426±0.0232 |
| `newG_cls` | 0.7543±0.0133 | 0.9416±0.0090 | 0.7536±0.0110 |
| `recon_newG` | 0.7725±0.0190 | 0.9544±0.0056 | 0.7692±0.0193 |

### 4.2 BlogCatalog simple baselines (P1; seeds 0–3, n=4)

Status **PASS** — 16/16 exit-0. Aggregate is **n=4** (seeds 0–3 only). Seed 42 was **not** rerun in P1; a separate seed-42 vanilla row exists in `sta_results.csv` (archival context, not part of the n=4 mean). Report: `priority1_blogcatalog_baselines.md`.

| Setting | Acc | AUC | macro-F1 | bal_acc | min_F1 | min_recall | PR-AUC | runtime_s |
|---------|----:|----:|---------:|--------:|-------:|-----------:|-------:|----------:|
| `no` | 0.1662±0.0114 | 0.5613±0.0121 | 0.0491±0.0096 | 0.0533±0.0078 | 0.0044±0.0048 | 0.0028±0.0031 | 0.0498±0.0044 | 676.5±99.3 |
| `upsampling` | 0.1666±0.0099 | 0.5632±0.0121 | 0.0517±0.0084 | 0.0551±0.0073 | 0.0108±0.0037 | 0.0090±0.0034 | 0.0498±0.0043 | 895.2±456.6 |
| `reweight` | 0.1664±0.0114 | 0.5618±0.0129 | 0.0533±0.0075 | 0.0568±0.0062 | 0.0149±0.0090 | 0.0127±0.0073 | 0.0495±0.0046 | 659.8±48.2 |
| `embed_up` | 0.1671±0.0108 | 0.5635±0.0137 | 0.0556±0.0072 | 0.0580±0.0063 | 0.0172±0.0073 | 0.0141±0.0047 | 0.0505±0.0046 | 875.8±419.8 |

#### Seed-42 BlogCatalog simple (STA CSV archival; not P1)

| Setting | Acc | AUC | macro-F1 |
|---------|----:|----:|---------:|
| `no` | 0.1798 | 0.5729 | 0.0566 |
| `upsampling` | 0.1739 | 0.5732 | 0.0579 |
| `reweight` | 0.1763 | 0.5724 | 0.0611 |
| `embed_up` | 0.1749 | 0.5729 | 0.0581 |

---

## 5. STA null ablation (matched Cora)

Source: `research/reports/sta_evaluation.md`, `research/results/sta_aggregate.md`. Same-Pretrained matched comparison (distinct from §4).

| Setting | Acc | AUC | macro-F1 |
|---------|----:|----:|---------:|
| `newG_cls` | 0.7273±0.0115 | 0.9381±0.0067 | 0.7243±0.0122 |
| `sta_newG_cls` | 0.7289±0.0121 | 0.9383±0.0067 | 0.7259±0.0134 |
| `recon_newG` | 0.7418±0.0106 | 0.9446±0.0051 | 0.7378±0.0126 |
| `sta_recon_newG` | 0.7392±0.0109 | 0.9457±0.0053 | 0.7350±0.0137 |

**Verdict:** STA does **not** produce a clear gain over matched GraphSMOTE on Cora (**null**). Soft attachment ≈ OR-template; `candidate=all` hurts. Keep as ablation only — not a positive method claim. **BlogCatalog STA / BlogCatalog STA-related simple seed-42 cells are seed-42 only** — not a multi-seed BlogCatalog STA claim.

---

## 6. Cora TAM sensitivity (OFAT; validation-only)

Status **PASS** — 50/50 exit-0. Primary tables **unchanged**. BlogCatalog sensitivity **not launched**. Sources: `priority2_tam_sensitivity.md`, `tam_sensitivity_selection.md`.

Locked primary mean **val** macro-F1 (n=5, primary TAM-on): **0.7347**.

| Factor | Level | mean val F1 | mean test F1 (post-hoc) | Decision |
|--------|------:|------------:|------------------------:|----------|
| `tam_alpha` | 1.0 | 0.7275 | 0.7413 | — |
| `tam_alpha` | 4.0 | 0.7368 | 0.7419 | **validation-only project sensitivity candidate** |
| `tam_beta` | 0.25 | 0.7337 | 0.7439 | — |
| `tam_beta` | 1.0 | 0.7339 | 0.7432 | keep primary |
| `tam_phi` | 1.0 | 0.7342 | 0.7432 | — |
| `tam_phi` | 1.5 | 0.7300 | 0.7396 | keep primary |
| `tam_delta` | 0.2 | 0.7347 | 0.7449 | — |
| `tam_delta` | 0.6 | 0.7347 | 0.7438 | keep primary |
| `tam_warmup` | 0 | 0.7360 | 0.7439 | **validation-only project sensitivity candidate** |
| `tam_warmup` | 20 | 0.7336 | 0.7445 | — |

α=4.0 and warmup=0 are **validation-only project sensitivity candidates** — appendix/future work only; **not** primary. Test metrics never used for selection. No multi-factor refit.

---

## 7. Competing methods (citation-only; not comparable)

**Not run** under GraphSMOTE-Plus fingerprint: GraphENS, GATSMOTE, GraphSHA, GraphSANN (`competing_methods_feasibility.md`). Do **not** paste their paper tables as our baselines. Related-work citations + “not reimplemented” limitation only.

| Method | Venue | Role |
|--------|-------|------|
| GraphENS | ICLR 2022 | Neighbor-aware ego synthesis; cite only |
| GATSMOTE | Mathematics 2022 | Attention/homophily synthesizer; cite only |
| GraphSHA | KDD 2023 | Harder minority synthesis; cite only |
| GraphSANN | IJCAI 2023 | Heterophily-aware synthesis; cite only |

---

## 8. Fair limitations

1. CPU / compatibility-adjusted PyTorch 2.5.1 — not bitwise Zhao/Song GPU tables; **runtime is descriptive CPU wall-clock** (§2.3), not a quality score.
2. Two datasets only; no CiteSeer/PubMed/heterophily suites.
3. Competitors (GraphENS/GATSMOTE/GraphSHA/GraphSANN) are **citation-only / not comparable** — **no SOTA claim**.
4. BlogCatalog simple baselines are **n=4** (seeds 0–3); seed 42 is archival-only, not in that mean.
5. BlogCatalog TAM sensitivity not run.
6. OFAT **validation-only project sensitivity candidates** not applied to primary.
7. Prefer **paired** matched Δ (§3) over absolute **non-matched** levels (§4).
8. STA multi-seed claim is Cora-matched only; **BlogCatalog STA remains seed-42 only**.
9. Cora Acc≈bal_acc is a **descriptive numerical observation** (§2.2), not an audited support-count claim.
10. Primary n=5 summaries are **descriptive**; **no significance testing** claimed.

---

## 9. Exact repro commands

```bash
cd /agent/repos/GraphSmote-Plus
source .venv/bin/activate
python research/tests/test_tam.py
python research/scripts/run_tam_eval.py --mode cora_primary --epochs 2010
python research/scripts/run_tam_eval.py --mode blog_tam --epochs 2010
python research/scripts/summarize_tam_results.py
python research/scripts/run_blog_simple_baselines.py --mode run --epochs 2010
python research/scripts/run_tam_sensitivity.py --mode run
python research/scripts/run_tam_sensitivity.py --mode select
python main.py --imbalance --no-cuda --dataset=cora --setting=newG_cls \
  --seed=42 --epochs=2010 --load=Pretrained --tam \
  --tam_alpha=2.5 --tam_beta=0.5 --tam_phi=1.2 --tam_delta=0.4 --tam_warmup=5 \
  --tam_scope=train_orig --config_out=research/configs/example_tam.json
```

---

## 10. Source index

| Artifact | Path |
|----------|------|
| TAM CSV / aggregate | `research/results/tam_results.csv`, `tam_aggregate.md` |
| Final audit / novelty | `/cursor/stores/self/docs/tam_final_audit.md`, `tam_novelty_review.md` |
| P0 / P1 / P2 | `research/reports/priority{0,1,2}_*.md` |
| STA | `research/reports/sta_evaluation.md`, `research/results/sta_aggregate.md` |
| Non-matched baselines | `research/reports/baseline_reproduction.md`, `baseline_results.csv` |
| Competitors feasibility | `research/competing_methods_feasibility.md` |
| Metrics / TAM impl | `utils.py` `print_class_acc`; `tam.py` |
| Sensitivity selection | `research/results/tam_sensitivity_selection.md` |

---

## 11. Draft title & claim boundaries

**Preferred working title:** *An Empirical Evaluation of Topology-Aware Margin Loss Integrated with GraphSMOTE for Imbalanced Node Classification*

**Allowed claims:** matched GS±TAM **descriptive** positive paired differences under **Song-reference settings plus explicitly documented integration settings**; STA null ablation on matched Cora; validation-only sensitivity candidates reported but not primary.

**Forbidden claims:** inventing TAM; “universal official TAM defaults”; treating δ as a Song-paper official named default without the integration caveat; SOTA vs unreimplemented competitors; STA as a win; multi-seed BlogCatalog STA; mixing §4 non-matched absolutes into TAM effect sizes; “beats/improves” without paired matched context; significance from n=5.

---

## 12. Validation appendix (wording pass)

| Check | Result |
|-------|--------|
| Primary cells exit-0 | **40/40** (unchanged; no CSV edits) |
| Numeric tables in this file | **Unchanged** vs prior evidence pack (wording-only) |
| CSV / code / logs / datasets / configs modified? | **No** |
| Experiments / BlogCatalog TAM sweep / competitor runs? | **None** |
| IEEE paper body edited? | **No** |
| Push/PR? | **No** |
