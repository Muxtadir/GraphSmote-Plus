# GraphSMOTE + TAM — Evaluation Report

**Date:** 2026-11-03  
**Branch:** `cursor/graphsmote-tam-cccf` (local; no push/PR)  
**Attribution:** GraphSMOTE = Zhao et al. (WSDM 2021); TAM = Song et al. (ICML 2022). Controlled hybrid — **not** a claim of inventing TAM.  
**STA:** remains documented null (`sta_evaluation.md`); preserved, not reclaimed.  
**Stack:** compatibility-adjusted PyTorch 2.5.1+cpu — not paper-table claims.  
**CSV:** `research/results/tam_results.csv`

---

## 1. Protocol

- Cora `im_ratio=0.5`, sage, seeds `{42,0,1,2,3}`, epochs `2010`.
- Matched finetunes from same per-seed `Pretrained.pth` (this VM).
- Primary: `newG_cls` ± TAM, `recon_newG` ± TAM.
- TAM defaults (official): α=2.5, β=0.5, φ=1.2, δ=0.4, warmup=5, scope=`train_orig`.
- **No retuning** after seeing results.

---

## 2. Cora matched results (n=5)

| Setting | TAM | Acc | AUC-ROC | macro-F1 |
|---------|-----|----:|--------:|---------:|
| newG_cls | off | 0.7283±0.0126 | 0.9395±0.0065 | 0.7251±0.0132 |
| newG_cls | **on** | 0.7470±0.0228 | **0.9472±0.0054** | **0.7449±0.0241** |
| recon_newG | off | 0.7314±0.0263 | 0.9410±0.0114 | 0.7281±0.0245 |
| recon_newG | **on** | 0.7501±0.0220 | **0.9464±0.0124** | **0.7481±0.0207** |

### Paired Δ = (GS+TAM) − GS

| Setting | mean ΔAUC | mean ΔF1 | Seeds with ΔF1>0 |
|---------|----------:|---------:|-----------------:|
| newG_cls | **+0.0077** | **+0.0198** | 5/5 |
| recon_newG | **+0.0054** | **+0.0199** | 5/5 |

**Cautious reading:** Under this locked CPU protocol, TAM improves GraphSMOTE finetune metrics with consistently positive paired deltas. This supports the hybrid as an empirical extension of Song et al.’s loss, not a new architecture claim. Variance on F1 remains non-trivial.

Figures: `research/figures/tam/cora_gs_vs_tam_auc_f1.png`, `cora_tam_paired_delta_f1.png`.

---

## 3. BlogCatalog

Seed-42 vanilla baselines already exist from the STA phase — **not re-run**. ETA logged in `research/logs/tam/blog_eta.txt` before launch.

### Seed 42 GraphSMOTE ± TAM (complete; preliminary single-seed)

| Setting | TAM | AUC-ROC | macro-F1 |
|---------|-----|--------:|---------:|
| newG_cls | off | 0.6139 | 0.0777 |
| newG_cls | on | **0.6215** | **0.0865** |
| recon_newG | off | 0.6276 | 0.1026 |
| recon_newG | on | **0.6465** | **0.1110** |

Paired Δ (seed42): newG_cls +0.0076 AUC / +0.0088 F1; recon_newG +0.0189 AUC / +0.0084 F1.  
**Do not** treat this as a multi-seed BlogCatalog final — seeds 0–3 continue / incomplete in runner.

---

## 4. Failed runs

Cora primary: all 20 runs exit 0 (no Traceback/NaN flags in runner).

---

## 5. Verdict

On matched Cora, **GraphSMOTE + TAM helps** vs GraphSMOTE alone (positive paired AUC/F1). STA remains a null attachment result. Recommended paper framing: cite both Zhao et al. and Song et al.; present TAM hybrid as loss-level topology correction complementary to GraphSMOTE synthesis.
