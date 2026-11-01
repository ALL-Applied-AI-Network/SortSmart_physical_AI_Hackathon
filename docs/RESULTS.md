# Quantitative results

## Fine-tuned YOLO vs baseline

Evaluation used five held-out robot-camera images with 40 annotated AAA battery instances. Predictions were matched to ground truth at the same evaluation settings.

| Metric | Baseline | Fine-tuned | Change |
|---|---:|---:|---:|
| True positives | 26 | 38 | +12 |
| False positives | 2 | 1 | -1 |
| False negatives | 14 | 2 | -12 |
| Precision | 92.9% | 97.4% | +4.5 pp |
| Recall | 65.0% | 95.0% | +30.0 pp |
| F1 score | 76.5% | 96.2% | +19.7 pp |
| Mean TP confidence | 73.8% | 82.7% | +8.9 pp |
| Image-cluster bootstrap 95% CI | 62.6–78.4% | 75.4–89.2% | — |

`Mean TP confidence` is the average model score among matched, correct battery detections. It is not a probability that a battery is a particular brand, and individual batteries receive different scores. The confidence intervals account for clustering by image, but the five images are from one held-out robot episode; additional independent scenes are needed for a stronger generalization estimate.

## Randomized digital-twin sweep

The v09 sweep varied brand, table position, yaw and wrapper roll.

| Measure | Result |
|---|---:|
| Overall end-to-end | 12/12 |
| Pickup verified | 12/12 |
| Carried to detected bowl | 12/12 |
| Final placement inside bowl | 12/12 |
| All quality gates passed | 12/12 |
| Amazon Basics | 6/6; mean confidence 0.8711 |
| 24/7 Life | 6/6; mean confidence 0.8646 |
| Primary/fallback camera selection | table 11; wrist 1 |

The observed success rate is 100%; the two-sided 95% Wilson interval is approximately 75.8–100%. The interval communicates the uncertainty from only 12 trials.

## Pose and depth checks

Representative perception reports showed sub-millimetre center error and roughly one-degree yaw error in the tested scenes. A stationary known-height-box diagnostic provided a real-cell depth sanity check:

| Check | Result |
|---|---:|
| Expected box height | 55–56 mm |
| Median measured height | 54.87 mm |
| Height standard deviation | 0.08 mm |
| Detection fraction | 100% across 20 frames |
| Object-height validation | passed |

These measurements validate the controlled setup, not unrestricted collision avoidance. Safe physical operation still requires calibrated frames, conservative margins, workspace limits and human supervision.

## Physical-robot outcome

The physical policy detected and localized the AAA battery, completed the grasp approach and lifted the object. The smooth stock fingers did not provide enough friction to retain the cylindrical battery. This is recorded as a manipulation-hardware limitation; it does not count as a successful physical pick-and-place trial.

