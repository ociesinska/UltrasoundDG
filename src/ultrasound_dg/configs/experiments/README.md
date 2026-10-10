# Experiment configurations

The final ERM reference is:

```text
baseline/baseline_multisource_tuned_v2.yaml
```

The directories separate active experiments by purpose:

- `baseline/` — the frozen reference configuration;
- `ablations/` — source-domain and sampling comparisons;
- `augmentations/` — experiments derived from the frozen baseline;
- `legacy/` — earlier V1 configurations retained for reproducibility.

Each reported experiment keeps its YAML because checkpoints and metrics alone
do not fully describe the split, preprocessing, optimizer, sampling, and model
selection settings. Generated configs under `outputs/` are runtime artifacts
and are not the canonical tracked definitions.
