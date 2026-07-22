# DeltaMLBench PWC Task Suitability Audit

## Purpose

This is a static first-pass audit of the 49 PWC task families exposed by the
Inspect runtime. Its main question is whether an agent has enough legitimate,
measurable room to improve on each baseline.

This pass uses the checked-in task instructions, manifests, build steps,
starter solutions, score implementations, and validation code. It does not
claim that a task is empirically validated. A task reaches `retain for pilot`
only when its static design justifies spending compute on a controlled run.

Implementation status: the 14 tasks classified `Remove` below have been moved
to `archive/old_tasks/` and removed from the Inspect registry. The active
surface is now 35 families and 70 variants.

## Recommendation summary

| Recommendation | Count | Meaning |
| --- | ---: | --- |
| Retain for pilot | 20 | Plausible headroom and a useful optimization surface; run before accepting. |
| Revise before pilot | 15 | Potentially useful, but the metric, baseline, objective, or evaluator needs work first. |
| Remove | 14 | Insufficient headroom or redundant coverage relative to a stronger retained task. |

This audit originally assumed independently recomputed metrics. The adopted
protocol instead accepts agent-reported metrics after a fail-closed code-and-log
integrity review, then compares them with the paper baseline. Recommendations
below that demand an independent evaluator are therefore historical hardening
suggestions rather than release requirements. The shared evaluation
issues below must be fixed before comparing agents.

## Shared blockers

### 1. The evaluator trusts agent-controlled evaluation

Every scorer imports `/home/agent/solution/solution.py`, calls its `evaluate()`
function, and scores the returned dictionary. The scorer does not own a hidden
test set, predictions, model loading, or metric recomputation. An agent can
change the split, evaluation population, preprocessing, or metric
implementation while still returning the expected key.

This makes a reported improvement difficult to distinguish from evaluation
drift, even when the solution is not explicitly fabricating a number.

Required fix: keep test labels and the metric implementation scorer-side. The
agent should submit a model, predictions, or an inference entry point that the
scorer evaluates independently.

### 2. Forensic grading is an availability-dependent score gate

All checked scorers enable forensic LLM grading by default. The forensic grader
returns `FAIL` when the corresponding API key is absent, and the scorer then
assigns zero. The sandbox compose file does not declare either provider key.
This means scoring depends on credential propagation and an external model,
not solely on the submitted solution.

Required fix: move forensic review out of the numeric scoring path, or make an
unavailable reviewer produce `not run` rather than `FAIL`. Pin and calibrate any
review model used for adjudication.

### 3. Artifact validation is mostly advisory

The task templates say that a model checkpoint and training log are required,
but missing artifacts are generally warnings rather than critical failures.
Conversely, fixed minimum checkpoint sizes and mandatory GPU evidence reject
legitimate small or CPU models and reward meaningless padding.

Required fix: define task-specific verifiable artifacts. Do not use file size
or GPU usage as a proxy for legitimate improvement.

### 4. Scores discard important objectives

Several tasks advertise multiple objectives but score only one:

- continual learning scores average accuracy but ignores backward transfer;
- Fashion-MNIST ENERGIZE scores accuracy but ignores power consumption;
- compact-model tasks ignore parameter counts;
- several forecasting tasks report both MSE and MAE but score only one;
- multiple representation-learning tasks report accuracy plus NMI but score
  only accuracy.

This invites an agent to sacrifice the actual paper contribution for the one
number that the benchmark rewards.

Required fix: enforce constraints or use a calibrated multi-objective score.

### 5. The baseline-returning starter is not a verified baseline

Most starter files are TODO templates that return a literal paper baseline.
They do not establish that the supplied code, data, dependencies, split, and
hardware reproduce that baseline. Some richer starters contain fallback paths
that return the documented value when execution fails, which the anti-cheat
code is intended to reject.

Required fix: each retained task needs a scorer-owned baseline run with a fixed
seed, recorded runtime, variance across runs, and saved predictions/artifacts.

## Decision rule

For metrics bounded by 1 or 100, `maximum relative score` is the score attainable
at the mathematical ceiling under the current formula. For example, a 97%
baseline can produce at most `(100 - 97) / 97 = 3.09%`, before accounting for
run-to-run noise. A small maximum does not prove that improvement is impossible,
but it makes the task a weak discriminator relative to its compute cost.

For unbounded error metrics such as MSE, MAE, and RMSE, static ceiling analysis
cannot establish practical headroom. These tasks are retained only for a pilot,
where the baseline must be reproduced and variance measured.

## Remove: insufficient scored headroom

| Task | Scored baseline | Maximum relative score | Decision evidence |
| --- | ---: | ---: | --- |
| `pwc_btad_urd` | detection AUROC 93.9 | 6.50% | Primary metric is already saturated; segmentation AUROC is 98.1 and the other segmentation metrics do not affect the score. |
| `pwc_cifar_10_abnet_2g_r0` | accuracy 94.118% | 6.25% | At most 5.882 accuracy points exist, and realistic gains occupy a small noisy slice of that range. |
| `pwc_cifar_10_resnet18_fsgdm` | accuracy 95.66% | 4.54% | Only 4.34 absolute points exist before the ceiling. |
| `pwc_clintox_bilstm` | ROC-AUC 0.97 | 3.09% | Only 0.03 AUC exists before the ceiling on a small, variance-sensitive molecular dataset. |
| `pwc_etth1_336_multivariate_softs` | MAE 0.452, MSE 0.480 | n/a | Duplicates AMD on the same ETTh1 multivariate 336-step protocol, while AMD has the stronger 0.427 MAE and 0.418 MSE baseline. |
| `pwc_kvasir_seg_effisegnet_b5` | mean Dice 0.9488; precision 0.9713 | about 6% across metrics | All five averaged segmentation metrics are close to their ceilings; small split or threshold changes can dominate genuine gains. |
| `pwc_kvasir_seg_emcad` | mean Dice 0.928 | 7.76% | Duplicates the retained Kvasir YOLO-SAM2 segmentation task, has less metric coverage, and lacked calibrated mIoU and boundary-IoU baselines. |
| `pwc_malnet_tiny_gatedgcn` | accuracy 94.6% | 5.71% | Narrow accuracy range relative to the training and graph-processing cost. |
| `pwc_mnist_gatedgcn` | accuracy 98.712% | 1.30% | Only 129 additional correct predictions per 10,000-example test set exist before perfect accuracy. |
| `pwc_mnist_rkan` | accuracy 99.293% | 0.71% | Only about 71 additional correct predictions exist before perfect accuracy; the starter fallback returns the paper value on failure. |
| `pwc_ogbl_ddi_gcn_node_embedding` | Hits@20 0.9549 | 4.72% | Metric is near saturation and the score ignores efficiency or model-size claims. |
| `pwc_stl_10_40_labels_semioccam` | accuracy 95.43% | 4.79% | Extremely high baseline for the 40-label setting leaves a narrow range likely dominated by split and seed variance. |
| `pwc_training_and_validation_dataset_of_capsule_vision_2024_challenge_biomedclip_pubmedbert` | accuracy 97.75% | 2.30% | Only 2.25 absolute points remain on a medical task where split integrity is especially important. |
| `pwc_txl_pbc_a_freely_accessible_labeled_peripheral_blood_cell_dataset_yolov5n` | mAP50 0.958 | 4.38% | Saturated mAP50 gives little discrimination and can hide localization-quality tradeoffs better shown by stricter mAP. |

These should not merely receive a rescaled score. The absolute opportunity for
improvement is narrow, and rescaling would mostly magnify evaluation noise.

## Revise before spending compute

| Task | Baseline | Required revision |
| --- | --- | --- |
| `pwc_5_datasets_code_cl` | average accuracy 93.32%, BWT -0.25 | Either remove or score backward transfer and resource cost alongside accuracy. The theoretical accuracy gain is only 6.68 points. |
| `pwc_california_housing_binary_diffusion` | MSE 0.39 | Redesign or remove. A scorer-owned pilot with an off-the-shelf `HistGradientBoostingRegressor` reached MSE 0.20647 (47.06% score), so the current objective rewards replacing Binary Diffusion with a generic regressor. |
| `pwc_cat2000_sum` | AUC 0.888, NSS 2.423 | Score NSS as well as AUC and verify that both metrics use a scorer-owned split. AUC alone has limited sensitivity here. |
| `pwc_cifar_100_pro_dsc` | accuracy 0.773, NMI 0.824 | Fix inconsistent percentage labeling and score the clustering objective (NMI), not only accuracy. |
| `pwc_cnn` | ROUGE-L 27.4 | Pin the summarization model and inference budget, and compute ROUGE scorer-side. Otherwise the task measures access to a stronger external model more than improvement to SigExt. |
| `pwc_electricity_192_cyclenet` | MSE 0.144, MAE 0.237 | Use both forecast metrics or justify one, then establish seeded baseline variance against the closely related forecasting tasks. |
| `pwc_fashion_mnist_continued_fraction_of_straight_lines` | accuracy 84.12%, NMI 74.4, 7,870 parameters | Enforce the parameter budget and include NMI; accuracy alone permits replacing the compact method with an unrelated large classifier. |
| `pwc_fashion_mnist_energize` | accuracy 90.2%, power 71.92 | Make power an enforced constraint or part of the score. Current scoring rewards accuracy regardless of the claimed energy objective. |
| `pwc_kvasir_seg_yolo_sam_2` | F1 90.7%, Dice 0.866, mIoU 0.764 | Normalize metric units and prevent averaging independent relative gains from masking regression on one metric. |
| `pwc_mm_vet_flashsloth_hd` | GPT-4 score 41.9, 3.2B parameters | Supply a deterministic evaluator and enforce the parameter/resource constraint. The starter has no MM-Vet evaluation implementation. |
| `pwc_office_31_euda` | accuracy 92.0% | Score per-domain adaptation performance and variance, not a single aggregate accuracy with only 8 points to the ceiling. |
| `pwc_stanford_cars_prometar` | harmonic mean 76.72% | Expose and constrain the seen/unseen components of the harmonic mean so an agent cannot improve one side by changing the evaluation balance. |
| `pwc_tiny_imagenet_classification_mano_tiny` | accuracy 87.52% | Enforce the “tiny” resource/parameter objective and verify the unusually strong baseline; otherwise larger pretrained models trivialize the task. |
| `pwc_wigesture_csi_bert` | accuracy 93.94% | Add subject/environment generalization metrics or remove. Plain accuracy has only 6.06 absolute points to the ceiling. |
| `pwc_zju_rgb_p_csfnet_2` | F1 90.7% | Add the complementary precision/recall or segmentation metrics and verify cross-subject/split behavior before accepting the remaining headroom. |

## Retain for controlled pilot

These have enough static headroom to justify a reproducibility and agent-improvement
trial. `Retain for pilot` is not a final keep decision.

| Task | Baseline | Why it merits a pilot | Principal pilot risk |
| --- | --- | --- | --- |
| `pwc_astock_srl_factors` | accuracy 69.48% | Large nominal classification headroom and several model/data levers. | Temporal leakage and unstable financial splits. |
| `pwc_chameleon_coed` | accuracy 79.69% | Material headroom on a manageable graph benchmark. | Fixed public split can be over-tuned. |
| `pwc_digital_twin_supported_deep_learning_for_fault_diagnosis_dann` | accuracy 80.22% | Meaningful accuracy range and domain-adaptation choices. | Domain split and synthetic/real leakage must be verified. |
| `pwc_etth1_336_multivariate_amd` | MAE 0.427 | Error metric has plausible room and forecasting offers real optimization choices. | Must reproduce the exact horizon, features, and split. |
| `pwc_etth1_720_multivariate_sparsetsf` | MSE 0.426 | Long-horizon setting can discriminate forecasting approaches. | Runtime and variance may swamp small gains. |
| `pwc_fashion_mnist_gecco` | accuracy 88.09% | Roughly 12 accuracy points remain and training is tractable. | Confirm the task is not trivialized by an unrestricted conventional CNN. |
| `pwc_fer2013_vgg_based` | accuracy 66.13% | Large nominal headroom and manageable classification workflow. | Label noise and split conventions can create fake gains. |
| `pwc_gowalla_rlae_dan` | Recall@20 0.1922 | Substantial ranking headroom and multiple legitimate recommender improvements. | Negative sampling and candidate-set definitions must be scorer-owned. |
| `pwc_mimic_iii_fld` | MSE 0.444 | Non-saturated error objective with model and preprocessing choices. | Patient leakage, access restrictions, and expensive preprocessing. |
| `pwc_ogbg_molhiv_gatedgcn` | ROC-AUC 0.774 | Meaningful metric range on a standard graph benchmark. | Official evaluator and split must be used scorer-side. |
| `pwc_pdbbind_bapulm` | RMSE 0.898 | Non-saturated regression objective with architectural headroom. | Protein-ligand split leakage can dominate results. |
| `pwc_pemsd4_pm_dmnet_r` | 12-step MAE 18.37 | Forecasting error has plausible room and several optimization levers. | Exact normalization and horizon definitions must be locked. |
| `pwc_peptides_struct_gcn` | MAE 0.2421 | Standard graph regression task with meaningful lower-is-better range. | Ensure official split/evaluator and compare with a simple strong baseline. |
| `pwc_summe_csta` | Kendall's tau 0.246 | Low baseline leaves substantial ranking headroom. | Small video dataset and subjective labels cause high variance. |
| `pwc_tiny_imagenet_pro_dsc` | accuracy 65.0% | Large accuracy range remains. | Compute cost and whether unrestricted pretrained models are allowed. |
| `pwc_traffic_glinear` | MSE 0.3222 | Non-saturated forecasting objective with tractable modifications. | Needs seed variance and consistent normalization. |
| `pwc_ucr_anomaly_archive_kan` | ROC-AUC 0.7489 | Material AUC headroom across anomaly-detection choices. | Archive aggregation can hide per-series regressions and leakage. |
| `pwc_weather_192_xpatch` | MAE 0.227 | Plausible error reduction and a clear time-series workflow. | Redundancy with other forecasting tasks and protocol sensitivity. |
| `pwc_york_urban_dataset_dt_lsd` | sAP10 33.2 | Large nominal detection headroom. | Dataset size, threshold sensitivity, and evaluator reproducibility. |
| `pwc_zinc_neuralwalker` | MAE 0.065 | Meaningful lower-is-better objective on a standard graph regression dataset. | Starter fallback and independent metric computation must be removed. |

## Pilot protocol

Run the 20 retained candidates in this order:

1. **Cheap calibration set:** Chameleon, Fashion-MNIST GECCO, FER2013, and
   ZINC. California Housing has already failed this stage because a generic
   boosted-tree baseline trivially beat the task baseline.
2. **Standard benchmark set:** ogbg-molhiv, Peptides-struct, ETTh1-336 AMD,
   Traffic, and UCR Anomaly Archive.
3. **Expensive or leakage-sensitive set:** the remaining forecasting, medical,
   finance, video, detection, and recommendation tasks.

For each pilot:

1. Replace self-reported evaluation with a scorer-owned hidden evaluation.
2. Run the checked-in baseline at least three times with fixed documented seeds.
3. Record setup time, training time, peak GPU memory, final metric, and variance.
4. Run a simple strong generic baseline, not just the paper implementation.
5. Give an agent at least two independent attempts under the intended budget.
6. Keep the task only if legitimate improvements exceed both baseline variance
   and a predeclared minimum practical effect.

A useful acceptance threshold is an improvement of at least two baseline
standard deviations and at least 1% relative improvement, with a higher floor
for expensive tasks. Tasks that a generic off-the-shelf model trivially beats
should be redesigned around constraints or removed rather than accepted as
research-engineering benchmarks.
