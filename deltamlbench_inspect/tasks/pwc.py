from __future__ import annotations

import json

from inspect_ai import Task, task
from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver._task_state import TaskState
from inspect_ai.util import sandbox

from deltamlbench_inspect import __version__
from deltamlbench_inspect.evaluation import evaluation_policy
from deltamlbench_inspect.integrity_judge import (
    apply_judge_verdict,
    run_integrity_judge,
    transcript_snapshot,
)
from deltamlbench_inspect.runtime import (
    SANDBOX_DOCKERFILE,
    discover_pwc_specs,
    parse_score_output,
    report_command,
    review_command,
    score_command,
    summarize_manifest_meta,
    task_sample,
)
from deltamlbench_inspect.solvers import modular_public_solver

@scorer(metrics=[mean(), stderr()])
def pwc_score(task_name: str, visible_score: bool = True):
    async def score(state: TaskState, target: Target) -> Score:
        del target
        agent_execution_events = transcript_snapshot()
        await sandbox().exec(
            cmd=["bash", "--login", "-c", report_command()],
            user="agent",
            timeout=7200,
        )
        result = await sandbox().exec(
            cmd=[
                "bash",
                "--login",
                "-c",
                score_command(True),
            ],
            user="root",
            timeout=7200,
        )
        output = f"{result.stderr or ''}\n{result.stdout or ''}".strip()
        try:
            payload = parse_score_output(output)
        except Exception as error:
            payload = {
                "score": 0.0,
                "status": "scoring_error",
                "message": {"error": f"Root scorer did not return a valid payload: {error}"},
            }
        if payload.get("status") == "scored":
            review_result = await sandbox().exec(
                cmd=["bash", "--login", "-c", review_command(task_name)],
                user="root",
                timeout=120,
            )
            try:
                review_bundle = json.loads(review_result.stdout or "")
                if not isinstance(review_bundle, dict):
                    raise ValueError("review bundle is not a JSON object")
                audit_for_judge = {
                    "status": payload.get("status"),
                    "reported_metrics": payload.get("reported_metrics", {}),
                    "measurements": payload.get("measurements", {}),
                    "constraint_failures": payload.get("constraint_failures", []),
                    "integrity": payload.get("integrity", {}),
                }
                judge = await run_integrity_judge(
                    task_instructions=state.input_text,
                    messages=list(state.messages),
                    review_bundle=review_bundle,
                    deterministic_audit=audit_for_judge,
                    execution_events=agent_execution_events,
                )
            except Exception as error:
                judge = {
                    "verdict": "error",
                    "confidence": 0.0,
                    "summary": f"Could not assemble review bundle: {error}",
                    "violations": [],
                    "mode": "error",
                }
            payload = apply_judge_verdict(payload, judge)
        else:
            payload["judge"] = {
                "verdict": "not_run",
                "confidence": 1.0,
                "summary": "Deterministic integrity or constraint checks did not pass.",
                "violations": [],
                "mode": "not_run",
            }
        value = payload.get("score", 0.0)
        return Score(
            value=float(value or 0.0),
            answer=state.output.completion or None,
            explanation=payload.get("message", {}).get("interpretation"),
            metadata=payload,
        )

    return score

_SPECS = {spec.name: spec for spec in discover_pwc_specs()}

def _build_named_task(task_name: str, variant_name: str) -> Task:
    spec = _SPECS[task_name]
    variant = next(variant for variant in spec.variants if variant.name == variant_name)
    task_meta = {
        "task_name": spec.name,
        "variant": variant.name,
        "title": spec.title,
        "manifest": summarize_manifest_meta(spec),
        "resources": spec.manifest.get("tasks", {}).get(variant.name, {}).get("resources", {}),
        "evaluation_policy": evaluation_policy(spec.name),
    }
    return Task(
        dataset=[task_sample(spec, variant)],
        solver=modular_public_solver(),
        scorer=pwc_score(spec.name, variant.visible_score),
        sandbox=("docker", SANDBOX_DOCKERFILE),
        time_limit=8 * 60 * 60,
        token_limit=10_000_000,
        metadata=task_meta,
        version=f"inspect-{__version__}",
        display_name=f"{spec.name}:{variant.name}",
    )

@task(name="pwc_5_datasets_code_cl_main", benchmark="deltamlbench", family="pwc", variant="main", provider="anthropic", gpu=True)
def pwc_5_datasets_code_cl_main() -> Task:
    return _build_named_task("pwc_5_datasets_code_cl", "main")

@task(name="pwc_5_datasets_code_cl_hidden_score", benchmark="deltamlbench", family="pwc", variant="hidden_score", provider="anthropic", gpu=True)
def pwc_5_datasets_code_cl_hidden_score() -> Task:
    return _build_named_task("pwc_5_datasets_code_cl", "hidden_score")

@task(name="pwc_astock_srl_factors_main", benchmark="deltamlbench", family="pwc", variant="main", provider="anthropic", gpu=True)
def pwc_astock_srl_factors_main() -> Task:
    return _build_named_task("pwc_astock_srl_factors", "main")

@task(name="pwc_astock_srl_factors_hidden_score", benchmark="deltamlbench", family="pwc", variant="hidden_score", provider="anthropic", gpu=True)
def pwc_astock_srl_factors_hidden_score() -> Task:
    return _build_named_task("pwc_astock_srl_factors", "hidden_score")

@task(name="pwc_california_housing_binary_diffusion_main", benchmark="deltamlbench", family="pwc", variant="main", provider="anthropic", gpu=True)
def pwc_california_housing_binary_diffusion_main() -> Task:
    return _build_named_task("pwc_california_housing_binary_diffusion", "main")

@task(name="pwc_california_housing_binary_diffusion_hidden_score", benchmark="deltamlbench", family="pwc", variant="hidden_score", provider="anthropic", gpu=True)
def pwc_california_housing_binary_diffusion_hidden_score() -> Task:
    return _build_named_task("pwc_california_housing_binary_diffusion", "hidden_score")

@task(name="pwc_cat2000_sum_main", benchmark="deltamlbench", family="pwc", variant="main", provider="anthropic", gpu=True)
def pwc_cat2000_sum_main() -> Task:
    return _build_named_task("pwc_cat2000_sum", "main")

@task(name="pwc_cat2000_sum_hidden_score", benchmark="deltamlbench", family="pwc", variant="hidden_score", provider="anthropic", gpu=True)
def pwc_cat2000_sum_hidden_score() -> Task:
    return _build_named_task("pwc_cat2000_sum", "hidden_score")

@task(name="pwc_chameleon_coed_main", benchmark="deltamlbench", family="pwc", variant="main", provider="anthropic", gpu=True)
def pwc_chameleon_coed_main() -> Task:
    return _build_named_task("pwc_chameleon_coed", "main")

@task(name="pwc_chameleon_coed_hidden_score", benchmark="deltamlbench", family="pwc", variant="hidden_score", provider="anthropic", gpu=True)
def pwc_chameleon_coed_hidden_score() -> Task:
    return _build_named_task("pwc_chameleon_coed", "hidden_score")

@task(name="pwc_cifar_100_pro_dsc_main", benchmark="deltamlbench", family="pwc", variant="main", provider="anthropic", gpu=True)
def pwc_cifar_100_pro_dsc_main() -> Task:
    return _build_named_task("pwc_cifar_100_pro_dsc", "main")

@task(name="pwc_cifar_100_pro_dsc_hidden_score", benchmark="deltamlbench", family="pwc", variant="hidden_score", provider="anthropic", gpu=True)
def pwc_cifar_100_pro_dsc_hidden_score() -> Task:
    return _build_named_task("pwc_cifar_100_pro_dsc", "hidden_score")

@task(name="pwc_cnn_main", benchmark="deltamlbench", family="pwc", variant="main", provider="anthropic", gpu=True)
def pwc_cnn_main() -> Task:
    return _build_named_task("pwc_cnn", "main")

@task(name="pwc_cnn_hidden_score", benchmark="deltamlbench", family="pwc", variant="hidden_score", provider="anthropic", gpu=True)
def pwc_cnn_hidden_score() -> Task:
    return _build_named_task("pwc_cnn", "hidden_score")

@task(name="pwc_digital_twin_supported_deep_learning_for_fault_diagnosis_dann_main", benchmark="deltamlbench", family="pwc", variant="main", provider="anthropic", gpu=True)
def pwc_digital_twin_supported_deep_learning_for_fault_diagnosis_dann_main() -> Task:
    return _build_named_task("pwc_digital_twin_supported_deep_learning_for_fault_diagnosis_dann", "main")

@task(name="pwc_digital_twin_supported_deep_learning_for_fault_diagnosis_dann_hidden_score", benchmark="deltamlbench", family="pwc", variant="hidden_score", provider="anthropic", gpu=True)
def pwc_digital_twin_supported_deep_learning_for_fault_diagnosis_dann_hidden_score() -> Task:
    return _build_named_task("pwc_digital_twin_supported_deep_learning_for_fault_diagnosis_dann", "hidden_score")

@task(name="pwc_electricity_192_cyclenet_main", benchmark="deltamlbench", family="pwc", variant="main", provider="anthropic", gpu=True)
def pwc_electricity_192_cyclenet_main() -> Task:
    return _build_named_task("pwc_electricity_192_cyclenet", "main")

@task(name="pwc_electricity_192_cyclenet_hidden_score", benchmark="deltamlbench", family="pwc", variant="hidden_score", provider="anthropic", gpu=True)
def pwc_electricity_192_cyclenet_hidden_score() -> Task:
    return _build_named_task("pwc_electricity_192_cyclenet", "hidden_score")

@task(name="pwc_etth1_336_multivariate_amd_main", benchmark="deltamlbench", family="pwc", variant="main", provider="anthropic", gpu=True)
def pwc_etth1_336_multivariate_amd_main() -> Task:
    return _build_named_task("pwc_etth1_336_multivariate_amd", "main")

@task(name="pwc_etth1_336_multivariate_amd_hidden_score", benchmark="deltamlbench", family="pwc", variant="hidden_score", provider="anthropic", gpu=True)
def pwc_etth1_336_multivariate_amd_hidden_score() -> Task:
    return _build_named_task("pwc_etth1_336_multivariate_amd", "hidden_score")

@task(name="pwc_etth1_720_multivariate_sparsetsf_main", benchmark="deltamlbench", family="pwc", variant="main", provider="anthropic", gpu=True)
def pwc_etth1_720_multivariate_sparsetsf_main() -> Task:
    return _build_named_task("pwc_etth1_720_multivariate_sparsetsf", "main")

@task(name="pwc_etth1_720_multivariate_sparsetsf_hidden_score", benchmark="deltamlbench", family="pwc", variant="hidden_score", provider="anthropic", gpu=True)
def pwc_etth1_720_multivariate_sparsetsf_hidden_score() -> Task:
    return _build_named_task("pwc_etth1_720_multivariate_sparsetsf", "hidden_score")

@task(name="pwc_fashion_mnist_continued_fraction_of_straight_lines_main", benchmark="deltamlbench", family="pwc", variant="main", provider="anthropic", gpu=True)
def pwc_fashion_mnist_continued_fraction_of_straight_lines_main() -> Task:
    return _build_named_task("pwc_fashion_mnist_continued_fraction_of_straight_lines", "main")

@task(name="pwc_fashion_mnist_continued_fraction_of_straight_lines_hidden_score", benchmark="deltamlbench", family="pwc", variant="hidden_score", provider="anthropic", gpu=True)
def pwc_fashion_mnist_continued_fraction_of_straight_lines_hidden_score() -> Task:
    return _build_named_task("pwc_fashion_mnist_continued_fraction_of_straight_lines", "hidden_score")

@task(name="pwc_fashion_mnist_energize_main", benchmark="deltamlbench", family="pwc", variant="main", provider="anthropic", gpu=True)
def pwc_fashion_mnist_energize_main() -> Task:
    return _build_named_task("pwc_fashion_mnist_energize", "main")

@task(name="pwc_fashion_mnist_energize_hidden_score", benchmark="deltamlbench", family="pwc", variant="hidden_score", provider="anthropic", gpu=True)
def pwc_fashion_mnist_energize_hidden_score() -> Task:
    return _build_named_task("pwc_fashion_mnist_energize", "hidden_score")

@task(name="pwc_fashion_mnist_gecco_main", benchmark="deltamlbench", family="pwc", variant="main", provider="anthropic", gpu=True)
def pwc_fashion_mnist_gecco_main() -> Task:
    return _build_named_task("pwc_fashion_mnist_gecco", "main")

@task(name="pwc_fashion_mnist_gecco_hidden_score", benchmark="deltamlbench", family="pwc", variant="hidden_score", provider="anthropic", gpu=True)
def pwc_fashion_mnist_gecco_hidden_score() -> Task:
    return _build_named_task("pwc_fashion_mnist_gecco", "hidden_score")

@task(name="pwc_fer2013_vgg_based_main", benchmark="deltamlbench", family="pwc", variant="main", provider="anthropic", gpu=True)
def pwc_fer2013_vgg_based_main() -> Task:
    return _build_named_task("pwc_fer2013_vgg_based", "main")

@task(name="pwc_fer2013_vgg_based_hidden_score", benchmark="deltamlbench", family="pwc", variant="hidden_score", provider="anthropic", gpu=True)
def pwc_fer2013_vgg_based_hidden_score() -> Task:
    return _build_named_task("pwc_fer2013_vgg_based", "hidden_score")

@task(name="pwc_gowalla_rlae_dan_main", benchmark="deltamlbench", family="pwc", variant="main", provider="anthropic", gpu=True)
def pwc_gowalla_rlae_dan_main() -> Task:
    return _build_named_task("pwc_gowalla_rlae_dan", "main")

@task(name="pwc_gowalla_rlae_dan_hidden_score", benchmark="deltamlbench", family="pwc", variant="hidden_score", provider="anthropic", gpu=True)
def pwc_gowalla_rlae_dan_hidden_score() -> Task:
    return _build_named_task("pwc_gowalla_rlae_dan", "hidden_score")

@task(name="pwc_kvasir_seg_yolo_sam_2_main", benchmark="deltamlbench", family="pwc", variant="main", provider="anthropic", gpu=True)
def pwc_kvasir_seg_yolo_sam_2_main() -> Task:
    return _build_named_task("pwc_kvasir_seg_yolo_sam_2", "main")

@task(name="pwc_kvasir_seg_yolo_sam_2_hidden_score", benchmark="deltamlbench", family="pwc", variant="hidden_score", provider="anthropic", gpu=True)
def pwc_kvasir_seg_yolo_sam_2_hidden_score() -> Task:
    return _build_named_task("pwc_kvasir_seg_yolo_sam_2", "hidden_score")

@task(name="pwc_mimic_iii_fld_main", benchmark="deltamlbench", family="pwc", variant="main", provider="anthropic", gpu=True)
def pwc_mimic_iii_fld_main() -> Task:
    return _build_named_task("pwc_mimic_iii_fld", "main")

@task(name="pwc_mimic_iii_fld_hidden_score", benchmark="deltamlbench", family="pwc", variant="hidden_score", provider="anthropic", gpu=True)
def pwc_mimic_iii_fld_hidden_score() -> Task:
    return _build_named_task("pwc_mimic_iii_fld", "hidden_score")

@task(name="pwc_mm_vet_flashsloth_hd_main", benchmark="deltamlbench", family="pwc", variant="main", provider="anthropic", gpu=True)
def pwc_mm_vet_flashsloth_hd_main() -> Task:
    return _build_named_task("pwc_mm_vet_flashsloth_hd", "main")

@task(name="pwc_mm_vet_flashsloth_hd_hidden_score", benchmark="deltamlbench", family="pwc", variant="hidden_score", provider="anthropic", gpu=True)
def pwc_mm_vet_flashsloth_hd_hidden_score() -> Task:
    return _build_named_task("pwc_mm_vet_flashsloth_hd", "hidden_score")

@task(name="pwc_office_31_euda_main", benchmark="deltamlbench", family="pwc", variant="main", provider="anthropic", gpu=True)
def pwc_office_31_euda_main() -> Task:
    return _build_named_task("pwc_office_31_euda", "main")

@task(name="pwc_office_31_euda_hidden_score", benchmark="deltamlbench", family="pwc", variant="hidden_score", provider="anthropic", gpu=True)
def pwc_office_31_euda_hidden_score() -> Task:
    return _build_named_task("pwc_office_31_euda", "hidden_score")

@task(name="pwc_ogbg_molhiv_gatedgcn_main", benchmark="deltamlbench", family="pwc", variant="main", provider="anthropic", gpu=True)
def pwc_ogbg_molhiv_gatedgcn_main() -> Task:
    return _build_named_task("pwc_ogbg_molhiv_gatedgcn", "main")

@task(name="pwc_ogbg_molhiv_gatedgcn_hidden_score", benchmark="deltamlbench", family="pwc", variant="hidden_score", provider="anthropic", gpu=True)
def pwc_ogbg_molhiv_gatedgcn_hidden_score() -> Task:
    return _build_named_task("pwc_ogbg_molhiv_gatedgcn", "hidden_score")

@task(name="pwc_pdbbind_bapulm_main", benchmark="deltamlbench", family="pwc", variant="main", provider="anthropic", gpu=True)
def pwc_pdbbind_bapulm_main() -> Task:
    return _build_named_task("pwc_pdbbind_bapulm", "main")

@task(name="pwc_pdbbind_bapulm_hidden_score", benchmark="deltamlbench", family="pwc", variant="hidden_score", provider="anthropic", gpu=True)
def pwc_pdbbind_bapulm_hidden_score() -> Task:
    return _build_named_task("pwc_pdbbind_bapulm", "hidden_score")

@task(name="pwc_pemsd4_pm_dmnet_r_main", benchmark="deltamlbench", family="pwc", variant="main", provider="anthropic", gpu=True)
def pwc_pemsd4_pm_dmnet_r_main() -> Task:
    return _build_named_task("pwc_pemsd4_pm_dmnet_r", "main")

@task(name="pwc_pemsd4_pm_dmnet_r_hidden_score", benchmark="deltamlbench", family="pwc", variant="hidden_score", provider="anthropic", gpu=True)
def pwc_pemsd4_pm_dmnet_r_hidden_score() -> Task:
    return _build_named_task("pwc_pemsd4_pm_dmnet_r", "hidden_score")

@task(name="pwc_peptides_struct_gcn_main", benchmark="deltamlbench", family="pwc", variant="main", provider="anthropic", gpu=True)
def pwc_peptides_struct_gcn_main() -> Task:
    return _build_named_task("pwc_peptides_struct_gcn", "main")

@task(name="pwc_peptides_struct_gcn_hidden_score", benchmark="deltamlbench", family="pwc", variant="hidden_score", provider="anthropic", gpu=True)
def pwc_peptides_struct_gcn_hidden_score() -> Task:
    return _build_named_task("pwc_peptides_struct_gcn", "hidden_score")

@task(name="pwc_stanford_cars_prometar_main", benchmark="deltamlbench", family="pwc", variant="main", provider="anthropic", gpu=True)
def pwc_stanford_cars_prometar_main() -> Task:
    return _build_named_task("pwc_stanford_cars_prometar", "main")

@task(name="pwc_stanford_cars_prometar_hidden_score", benchmark="deltamlbench", family="pwc", variant="hidden_score", provider="anthropic", gpu=True)
def pwc_stanford_cars_prometar_hidden_score() -> Task:
    return _build_named_task("pwc_stanford_cars_prometar", "hidden_score")

@task(name="pwc_summe_csta_main", benchmark="deltamlbench", family="pwc", variant="main", provider="anthropic", gpu=True)
def pwc_summe_csta_main() -> Task:
    return _build_named_task("pwc_summe_csta", "main")

@task(name="pwc_summe_csta_hidden_score", benchmark="deltamlbench", family="pwc", variant="hidden_score", provider="anthropic", gpu=True)
def pwc_summe_csta_hidden_score() -> Task:
    return _build_named_task("pwc_summe_csta", "hidden_score")

@task(name="pwc_tiny_imagenet_classification_mano_tiny_main", benchmark="deltamlbench", family="pwc", variant="main", provider="anthropic", gpu=True)
def pwc_tiny_imagenet_classification_mano_tiny_main() -> Task:
    return _build_named_task("pwc_tiny_imagenet_classification_mano_tiny", "main")

@task(name="pwc_tiny_imagenet_classification_mano_tiny_hidden_score", benchmark="deltamlbench", family="pwc", variant="hidden_score", provider="anthropic", gpu=True)
def pwc_tiny_imagenet_classification_mano_tiny_hidden_score() -> Task:
    return _build_named_task("pwc_tiny_imagenet_classification_mano_tiny", "hidden_score")

@task(name="pwc_tiny_imagenet_pro_dsc_main", benchmark="deltamlbench", family="pwc", variant="main", provider="anthropic", gpu=True)
def pwc_tiny_imagenet_pro_dsc_main() -> Task:
    return _build_named_task("pwc_tiny_imagenet_pro_dsc", "main")

@task(name="pwc_tiny_imagenet_pro_dsc_hidden_score", benchmark="deltamlbench", family="pwc", variant="hidden_score", provider="anthropic", gpu=True)
def pwc_tiny_imagenet_pro_dsc_hidden_score() -> Task:
    return _build_named_task("pwc_tiny_imagenet_pro_dsc", "hidden_score")

@task(name="pwc_traffic_glinear_main", benchmark="deltamlbench", family="pwc", variant="main", provider="anthropic", gpu=True)
def pwc_traffic_glinear_main() -> Task:
    return _build_named_task("pwc_traffic_glinear", "main")

@task(name="pwc_traffic_glinear_hidden_score", benchmark="deltamlbench", family="pwc", variant="hidden_score", provider="anthropic", gpu=True)
def pwc_traffic_glinear_hidden_score() -> Task:
    return _build_named_task("pwc_traffic_glinear", "hidden_score")

@task(name="pwc_ucr_anomaly_archive_kan_main", benchmark="deltamlbench", family="pwc", variant="main", provider="anthropic", gpu=True)
def pwc_ucr_anomaly_archive_kan_main() -> Task:
    return _build_named_task("pwc_ucr_anomaly_archive_kan", "main")

@task(name="pwc_ucr_anomaly_archive_kan_hidden_score", benchmark="deltamlbench", family="pwc", variant="hidden_score", provider="anthropic", gpu=True)
def pwc_ucr_anomaly_archive_kan_hidden_score() -> Task:
    return _build_named_task("pwc_ucr_anomaly_archive_kan", "hidden_score")

@task(name="pwc_weather_192_xpatch_main", benchmark="deltamlbench", family="pwc", variant="main", provider="anthropic", gpu=True)
def pwc_weather_192_xpatch_main() -> Task:
    return _build_named_task("pwc_weather_192_xpatch", "main")

@task(name="pwc_weather_192_xpatch_hidden_score", benchmark="deltamlbench", family="pwc", variant="hidden_score", provider="anthropic", gpu=True)
def pwc_weather_192_xpatch_hidden_score() -> Task:
    return _build_named_task("pwc_weather_192_xpatch", "hidden_score")

@task(name="pwc_wigesture_csi_bert_main", benchmark="deltamlbench", family="pwc", variant="main", provider="anthropic", gpu=True)
def pwc_wigesture_csi_bert_main() -> Task:
    return _build_named_task("pwc_wigesture_csi_bert", "main")

@task(name="pwc_wigesture_csi_bert_hidden_score", benchmark="deltamlbench", family="pwc", variant="hidden_score", provider="anthropic", gpu=True)
def pwc_wigesture_csi_bert_hidden_score() -> Task:
    return _build_named_task("pwc_wigesture_csi_bert", "hidden_score")

@task(name="pwc_york_urban_dataset_dt_lsd_main", benchmark="deltamlbench", family="pwc", variant="main", provider="anthropic", gpu=True)
def pwc_york_urban_dataset_dt_lsd_main() -> Task:
    return _build_named_task("pwc_york_urban_dataset_dt_lsd", "main")

@task(name="pwc_york_urban_dataset_dt_lsd_hidden_score", benchmark="deltamlbench", family="pwc", variant="hidden_score", provider="anthropic", gpu=True)
def pwc_york_urban_dataset_dt_lsd_hidden_score() -> Task:
    return _build_named_task("pwc_york_urban_dataset_dt_lsd", "hidden_score")

@task(name="pwc_zinc_neuralwalker_main", benchmark="deltamlbench", family="pwc", variant="main", provider="anthropic", gpu=True)
def pwc_zinc_neuralwalker_main() -> Task:
    return _build_named_task("pwc_zinc_neuralwalker", "main")

@task(name="pwc_zinc_neuralwalker_hidden_score", benchmark="deltamlbench", family="pwc", variant="hidden_score", provider="anthropic", gpu=True)
def pwc_zinc_neuralwalker_hidden_score() -> Task:
    return _build_named_task("pwc_zinc_neuralwalker", "hidden_score")

@task(name="pwc_zju_rgb_p_csfnet_2_main", benchmark="deltamlbench", family="pwc", variant="main", provider="anthropic", gpu=True)
def pwc_zju_rgb_p_csfnet_2_main() -> Task:
    return _build_named_task("pwc_zju_rgb_p_csfnet_2", "main")

@task(name="pwc_zju_rgb_p_csfnet_2_hidden_score", benchmark="deltamlbench", family="pwc", variant="hidden_score", provider="anthropic", gpu=True)
def pwc_zju_rgb_p_csfnet_2_hidden_score() -> Task:
    return _build_named_task("pwc_zju_rgb_p_csfnet_2", "hidden_score")
