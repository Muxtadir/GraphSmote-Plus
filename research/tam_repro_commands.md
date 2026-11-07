# GraphSMOTE + TAM — Repro Commands

CPU-only. Attribution: GraphSMOTE (Zhao et al., WSDM 2021); TAM (Song et al., ICML 2022).

```bash
cd /agent/repos/GraphSmote-Plus
source .venv/bin/activate

# Unit tests
python research/tests/test_tam.py

# Smoke (5 epochs): TAM off vs on
python research/scripts/run_tam_eval.py --mode smoke --seed 42 --epochs 5

# ETA estimate
python research/scripts/run_tam_eval.py --mode eta --epochs 2010

# Cora primary matched suite (seeds 42,0,1,2,3)
python research/scripts/run_tam_eval.py --mode cora_primary --epochs 2010

# BlogCatalog GS±TAM (does NOT re-run seed42 vanilla baselines)
python research/scripts/run_tam_eval.py --mode blog_tam --epochs 2010

# Manual example
python main.py --imbalance --no-cuda --dataset=cora --setting=newG_cls \
  --seed=42 --epochs=2010 --load=Pretrained --tam \
  --config_out=research/configs/example_tam.json
```

Resume: runner skips `run_id` already present with `exit_code=0` in `research/results/tam_results.csv` unless `--force`.
