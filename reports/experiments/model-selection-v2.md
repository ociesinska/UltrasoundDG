# Model selection and tuning V2

V1 selected the checkpoint with the highest macro source lesion Dice. The domain-balanced experiment showed that models with similar lesion Dice can behave very differently on normal scans, so V2 also considers false-positive area. This is a revised selection protocol for the same ERM baseline, not a new model.

## Checkpoint selection

For every epoch, the pipeline calculates macro source lesion Dice and mean false-positive fraction on normal source-validation images. It then:

1. retains checkpoints within `0.01` of the best macro lesion Dice;
2. selects the one with the lowest normal false-positive fraction;
3. uses higher lesion Dice to break an exact tie.

If source validation contains no normal cases, selection falls back to the highest macro lesion Dice. Only source-validation metrics are used; BUS-UCLM and BrEaST do not influence checkpoint selection.

## Hyperparameter tuning

Optuna still uses macro lesion Dice for pruning. Each trial selects its best epoch with the rule above, and the same rule is applied across completed trials. The selected checkpoint and complete experiment configuration are saved for direct evaluation.

In the multi-source protocol, normal source-validation cases come only from BUSI. This makes the secondary metric relatively noisy, but the `0.01` tolerance prevents it from overriding a meaningful Dice difference.

The tuning configuration is stored in
`src/ultrasound_dg/configs/tuning/baseline_v2.yaml`. The completed search
selected AdamW with learning rate `1e-4` and weight decay `1e-3`. The resulting
natural multi-source model was evaluated with three training seeds and is now
the final ERM reference. Its configuration is stored in
`src/ultrasound_dg/configs/experiments/baseline/baseline_multisource_tuned_v2.yaml`;
results are reported in [Final ERM baseline](../results/baseline.md).

V1 configurations and results are retained only as development history. New
experiments use V2 selection and compare against the three-seed V2 baseline.
