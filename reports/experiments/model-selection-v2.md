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

The configuration is stored in `src/ultrasound_dg/configs/tuning/baseline_v2.yaml`. V1 results remain preliminary evidence; V2 becomes the primary baseline protocol after the relevant experiments are rerun.
