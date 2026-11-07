# GraphSMOTE + TAM — Implementation Notes

**Date:** 2026-11-03  
**Branch:** `cursor/graphsmote-tam-cccf`  
**Design:** `research/tam_design.md`  
**Attribution:** GraphSMOTE = Zhao et al. WSDM 2021; TAM = Song et al. ICML 2022 (official code `Jaeyun-Song/TAM`). Hybrid study — not a novelty claim for inventing TAM.

---

## Files

| Path | Role |
|------|------|
| `tam.py` | Pure-PyTorch ACM/ADM + logit adjust (no torch_geometric) |
| `utils.py` | CLI `--tam*` flags; config logging |
| `main.py` | CE path uses adjusted logits when `--tam` and not `recon` |
| `research/tests/test_tam.py` | Unit tests |
| `research/scripts/run_tam_eval.py` | Evaluation runner |

## Behavior

- `--tam` **off** (default): identical CE to GraphSMOTE.
- `--tam` **on**: after classifier forward, adjust logits for original train nodes via ACM/ADM; CE on adjusted logits; val/test prints use raw logits.
- Skipped entirely for `setting=recon` (decoder pretrain).
- Topology for NLD: **original** adjacency only.
- Margins computed under `@torch.no_grad()` (official).

## Defaults

`tam_alpha=2.5`, `tam_beta=0.5`, `tam_phi=1.2`, `tam_delta=0.4`, `tam_warmup=5`, `tam_scope=train_orig`.

## Deps

No new packages. Mean aggregation reimplemented vs official PyG `MeanAggregation`.
