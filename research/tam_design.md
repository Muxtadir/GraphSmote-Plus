# GraphSMOTE + TAM — Design (controlled hybrid)

**Date:** 2026-11-03  
**Branch:** `cursor/graphsmote-tam-cccf`  
**Status:** Design frozen before coding.  
**Attribution:**  
- GraphSMOTE = Zhao, Zhang, Wang (WSDM 2021).  
- TAM = Song, Park, Yang (ICML 2022).  
- STA = GraphSMOTE-Plus v1 (documented **null** on matched Cora); preserved, not reclaimed.  
**Novelty stance:** This is a **controlled hybrid / empirical extension**. TAM is prior work. Combining GraphSMOTE with TAM does **not** invent TAM and is **not** automatic novelty.

**Store mirror:** `/cursor/stores/self/docs/tam_design.md`

---

## 1. Why TAM (after STA null)

STA improved synthetic **edge attachment** and did not beat GraphSMOTE under matched protocol (`sta_evaluation.md`). Ranked alternative #2 from the Phase-7 design was **GraphSMOTE + TAM**: a **loss-side** topology-aware margin that addresses a different failure mode — quantity-only imbalance methods raising minority TPR while inflating **false positives** near anomalous connectivity (Song et al., ICML 2022).

TAM is the smallest meaningful next study: keep GraphSMOTE’s encoder/SMOTE/decoder unchanged; change only the **classifier CE logits** for labeled train nodes.

---

## 2. What stays GraphSMOTE

Unchanged when `--tam` is off **and** when TAM is on (except the CE logits path):

- Encoder / classifier / decoder architectures (`models.py`).
- `recon_upsample` SMOTE + OR-template (and STA flags if used).
- Decoder recon loss on the original \(N\times N\) block.
- Train/val/test splits (`split_arti` / `split_genuine`).
- Settings `no`, `upsampling`, `reweight`, `embed_up`, `recon`, `newG_cls`, `recon_newG`, `sta_*`.
- Default epochs, lr, dropout, `up_scale`, `im_ratio`, seeds protocol.

**TAM off ⇒ bit-compatible training objective with current GraphSMOTE CE** (same `F.cross_entropy` call sites).

---

## 3. Official TAM sources (read before coding)

| Source | Location |
|--------|----------|
| Paper | Song, Park, Yang. *TAM: Topology-Aware Margin Loss for Class-Imbalanced Node Classification.* ICML 2022, PMLR 162:20369–20383. https://proceedings.mlr.press/v162/song22a.html ; PDF: `.../song22a/song22a.pdf` |
| Official code | https://github.com/Jaeyun-Song/TAM (MIT); cloned locally to `/tmp/tam_ref/TAM` for inspection |
| Core module | `models/tam.py`: `compute_tam`, `adjust_output`, `MeanAggregation` |
| Defaults (`args.py`) | `tam_alpha=2.5`, `tam_beta=0.5`, `temp_phi=1.2`, `warmup=5`; `temp_gamma=0.4` hardcoded in `adjust_output` |

### 3.1 Equations (as implemented officially)

Neighbor label distribution (NLD) for node \(v\): mean-aggregate class distributions over neighbors (with self-loop). Labeled train nodes contribute **one-hot** of \(y_v\); unlabeled nodes contribute **softmax**(logits / class-wise temperature). Class-wise temperature uses \(\phi\) (`temp_phi`) and \(\gamma=0.4\).

**ACM** (Anomalous Connectivity Margin): compares node NLD vs class-averaged connectivity matrix; values clipped to \(\le 1\); true-class entry forced to 1. Applied as \(\alpha \log(\mathrm{ACM})\).

**ADM** (Anomalous Distribution Margin): JSD-based deviation of node NLD from class averages (law of cosines form in code). True-class ADM forced to 0. Applied as \(\beta (-\mathrm{ADM})\).

**Loss:**
\[
\mathcal{L}_{\mathrm{TAM}} = \frac{1}{|V^L|}\sum_{v\in V^L} \ell\!\big(\mathrm{logit}_v + \alpha\, m^{\mathrm{ACM}}_v + \beta\, m^{\mathrm{ADM}}_v,\, y_v\big)
\]
with \(m^{\mathrm{ACM}}=\log(\mathrm{ACM})\), \(m^{\mathrm{ADM}}=-\mathrm{ADM}\), and \(\ell=\) CE (or whatever base loss; we use CE to match GraphSMOTE).

`compute_tam` is `@torch.no_grad()` — margins **detached** from the graph/prediction path used to build NLD (official).

---

## 4. Insertion point in GraphSMOTE

**Where:** `main.py` `train()`, immediately after `output = classifier(...)`, when building `loss_train`.

**How:**
1. If `--tam` and `setting != 'recon'` and `epoch > tam_warmup`: compute ACM/ADM from **original-graph** topology + current logits; adjust logits on scoped labeled nodes; CE on adjusted logits.
2. Else: existing `F.cross_entropy(output[idx_train_new], labels_new[idx_train_new])` unchanged.

**Not during** decoder-only pretrain (`setting=recon`): CE is already zeroed; TAM must not alter recon dynamics.

**Val/test:** never use val/test labels for NLD one-hots, class counts, ACM/ADM, or HP selection beyond reporting. Eval remains plain logits (official `test()` also uses unadjusted logits).

---

## 5. Which nodes get margins (`--tam_scope`)

| Scope | Meaning |
|-------|---------|
| `train_orig` (**default**) | Original labeled **train** nodes only (`idx_train`). Matches paper \(V^L\). Synthetics excluded from TAM adjustment. |
| `train_all` | All nodes in `idx_train_new` (includes GraphSMOTE synthetics). Ablation only. |

Default hybrid: GraphSMOTE still trains CE on its usual index set (`idx_train` for `newG_cls`; `idx_train_new` for `recon_newG`), but **margins are computed/applied only on original train nodes** when scope=`train_orig`. For synthetic indices under `newG_cls` there are none in CE; under `recon_newG`, synthetics keep **unadjusted** CE logits unless scope=`train_all`.

---

## 6. Neighbor-label matrix \(D\) / NLD

- Topology: **original** adjacency (pre-augmentation), converted to edge list or dense mean-agg with self-loops.
- For each neighbor \(u\): if \(u\in\) train labeled set → one-hot \(y_u\); else → temperature-scaled softmax of current logits at \(u\).
- **Unlabeled** includes val/test **structure** (transductive) but **not** their gold labels — same as official TAM (`agg_out[train_mask]=one_hot`).
- Class-averaged connectivity matrix: mean NLD over train nodes of each class.

---

## 7. Synthetics

- Default: **do not** treat synthetics as labeled centers for ACM/ADM; do not require synthetic rows in the original graph.
- GraphSMOTE synth/decoder path unchanged.
- Optional `train_all`: map synthetic nodes’ neighbors via the **augmented** graph only if explicitly enabled later; v1 default keeps topology = original graph and scope = `train_orig` to avoid confounding STA/edge generation with TAM.

---

## 8. Leakage prevention

Forbidden:
- Val/test gold labels in NLD one-hots, class_num_list, or margin formulas.
- Tuning α/β/φ/δ/warmup on test.
- Silently changing epochs/lr/seeds vs GraphSMOTE matched protocol.

Allowed:
- Original undirected edges involving all nodes (transductive).
- Train labels + model predictions for unlabeled nodes.
- Val metrics for optional future HP selection (not used in default table — defaults pre-registered from official TAM).

---

## 9. Hyperparameters (CLI)

| Flag | Default | Maps to official |
|------|---------|------------------|
| `--tam` | off | `--tam` |
| `--tam_alpha` | 2.5 | `tam_alpha` (ACM strength) |
| `--tam_beta` | 0.5 | `tam_beta` (ADM strength) |
| `--tam_phi` | 1.2 | `temp_phi` |
| `--tam_delta` | 0.4 | `temp_gamma` (official hardcoded 0.4) |
| `--tam_warmup` | 5 | `warmup` (epochs; TAM active when `epoch > warmup`) |
| `--tam_scope` | `train_orig` | (GraphSMOTE-Plus only) |

No new pip packages: reimplement mean aggregation in pure PyTorch (official uses `torch_geometric` / `torch_scatter` — **not** added here to keep the CPU repro stack). Documented deviation: same math, different sparse backend.

---

## 10. Experiments (planned)

**Primary (Cora, `im_ratio=0.5`, sage, seeds 42,0,1,2,3, 2010 epochs):**
1. `newG_cls`
2. `newG_cls --tam`
3. `recon_newG`
4. `recon_newG --tam`

Prefer **matched** finetunes from same per-seed `Pretrained.pth` on this VM. Reuse prior matched GS rows from `sta_results.csv` only when protocol/VM-identical; otherwise re-run.

**Optional if cheap:** `no` ± TAM (short or full).

**BlogCatalog:** Seed-42 baselines already exist from STA worker — **do not duplicate**. Run GraphSMOTE+TAM (`newG_cls`/`recon_newG` + `--tam`) with same seeds; log ETA; resume safely; do not present partial BlogCatalog as final multi-seed claim.

**Metrics:** acc, AUC-ROC, macro-F1, minority F1/recall, balanced acc, PR-AUC, runtime; **paired** Δ = (GS+TAM) − GS per seed.

**Negative rule:** If TAM does not improve under matched protocol, report honest negative; **do not retune until win**.

---

## 11. Hybrid-not-novelty statement

This work studies whether **Song et al.’s TAM**, applied as a drop-in logit adjustment on GraphSMOTE finetune, improves imbalance metrics under our locked reproduction protocol. Credit for ACM/ADM belongs to Song et al. (ICML 2022). GraphSMOTE remains Zhao et al. (WSDM 2021). Results are compatibility-adjusted (PyTorch 2.5.1+cpu), not paper-table claims.

---

## 12. Implementation checklist (next phases)

1. `utils.py` / `tam_loss.py`: pure-torch `compute_tam` + `adjust_logits_tam`; CLI flags; config logging.
2. `main.py`: wire TAM into CE path; skip on `recon`; TAM-off path identical.
3. `research/tests/test_tam.py` + Cora smoke off/on.
4. `research/scripts/run_tam_eval.py`; CSV/aggregate/figures; store copies.
