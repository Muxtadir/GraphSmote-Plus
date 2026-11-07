# TAM aggregate (recomputed from tam_results.csv)

Attribution: GraphSMOTE=Zhao et al. WSDM 2021; TAM=Song et al. ICML 2022 (hybrid).

| Dataset | Setting | TAM | n | Acc mean±std | AUC mean±std | macro-F1 mean±std |
|---------|---------|-----|--:|-------------:|-------------:|------------------:|
| BlogCatalog | `newG_cls` | False | 1 | 0.1980±0.0000 | 0.6139±0.0000 | 0.0777±0.0000 |
| BlogCatalog | `newG_cls` | True | 1 | 0.2037±0.0000 | 0.6215±0.0000 | 0.0865±0.0000 |
| BlogCatalog | `recon_newG` | False | 1 | 0.2377±0.0000 | 0.6276±0.0000 | 0.1026±0.0000 |
| BlogCatalog | `recon_newG` | True | 1 | 0.2503±0.0000 | 0.6465±0.0000 | 0.1110±0.0000 |
| cora | `newG_cls` | False | 5 | 0.7283±0.0126 | 0.9395±0.0065 | 0.7251±0.0132 |
| cora | `newG_cls` | True | 5 | 0.7470±0.0228 | 0.9472±0.0054 | 0.7449±0.0241 |
| cora | `recon_newG` | False | 5 | 0.7314±0.0263 | 0.9410±0.0114 | 0.7281±0.0245 |
| cora | `recon_newG` | True | 5 | 0.7501±0.0220 | 0.9464±0.0124 | 0.7481±0.0207 |

## Paired Δ = (GS+TAM) − GS (Cora)

| Setting | Seed | Δ AUC | Δ macro-F1 |
|---------|-----:|------:|-----------:|
| `newG_cls` | 42 | +0.0024 | +0.0050 |
| `newG_cls` | 0 | +0.0103 | +0.0349 |
| `newG_cls` | 1 | +0.0071 | +0.0456 |
| `newG_cls` | 2 | +0.0104 | +0.0096 |
| `newG_cls` | 3 | +0.0082 | +0.0041 |
| `newG_cls` | mean | +0.0077 | +0.0198 |
| `recon_newG` | 42 | +0.0024 | +0.0222 |
| `recon_newG` | 0 | +0.0077 | +0.0309 |
| `recon_newG` | 1 | +0.0033 | +0.0170 |
| `recon_newG` | 2 | +0.0089 | +0.0089 |
| `recon_newG` | 3 | +0.0049 | +0.0208 |
| `recon_newG` | mean | +0.0054 | +0.0199 |
