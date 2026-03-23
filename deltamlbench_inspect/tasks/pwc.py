from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver._task_state import TaskState
from inspect_ai.util import sandbox

from deltamlbench_inspect.runtime import (
    SANDBOX_DOCKERFILE,
    discover_pwc_specs,
    parse_score_output,
    score_command,
    summarize_manifest_meta,
    task_sample,
)
from deltamlbench_inspect.solvers import modular_public_solver

@scorer(metrics=[mean(), stderr()])
def pwc_score(visible_score: bool = True):
    async def score(state: TaskState, target: Target) -> Score:
        del target
        result = await sandbox().exec(
            cmd=["bash", "--login", "-c", score_command(visible_score)],
            timeout=7200,
        )
        output = f"{result.stderr or ''}\n{result.stdout or ''}".strip()
        payload = parse_score_output(output)
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
    }
    return Task(
        dataset=[task_sample(spec, variant)],
        solver=modular_public_solver(),
        scorer=pwc_score(variant.visible_score),
        sandbox=("docker", SANDBOX_DOCKERFILE),
        time_limit=8 * 60 * 60,
        token_limit=10_000_000,
        metadata=task_meta,
        version="inspect-v1",
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

@task(name="pwc_btad_urd_main", benchmark="deltamlbench", family="pwc", variant="main", provider="anthropic", gpu=True)
def pwc_btad_urd_main() -> Task:
    return _build_named_task("pwc_btad_urd", "main")

@task(name="pwc_btad_urd_hidden_score", benchmark="deltamlbench", family="pwc", variant="hidden_score", provider="anthropic", gpu=True)
def pwc_btad_urd_hidden_score() -> Task:
    return _build_named_task("pwc_btad_urd", "hidden_score")

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

@task(name="pwc_cifar_10_abnet_2g_r0_main", benchmark="deltamlbench", family="pwc", variant="main", provider="anthropic", gpu=True)
def pwc_cifar_10_abnet_2g_r0_main() -> Task:
    return _build_named_task("pwc_cifar_10_abnet_2g_r0", "main")

@task(name="pwc_cifar_10_abnet_2g_r0_hidden_score", benchmark="deltamlbench", family="pwc", variant="hidden_score", provider="anthropic", gpu=True)
def pwc_cifar_10_abnet_2g_r0_hidden_score() -> Task:
    return _build_named_task("pwc_cifar_10_abnet_2g_r0", "hidden_score")

@task(name="pwc_cifar_10_resnet18_fsgdm_main", benchmark="deltamlbench", family="pwc", variant="main", provider="anthropic", gpu=True)
def pwc_cifar_10_resnet18_fsgdm_main() -> Task:
    return _build_named_task("pwc_cifar_10_resnet18_fsgdm", "main")

@task(name="pwc_cifar_10_resnet18_fsgdm_hidden_score", benchmark="deltamlbench", family="pwc", variant="hidden_score", provider="anthropic", gpu=True)
def pwc_cifar_10_resnet18_fsgdm_hidden_score() -> Task:
    return _build_named_task("pwc_cifar_10_resnet18_fsgdm", "hidden_score")

@task(name="pwc_clintox_bilstm_main", benchmark="deltamlbench", family="pwc", variant="main", provider="anthropic", gpu=True)
def pwc_clintox_bilstm_main() -> Task:
    return _build_named_task("pwc_clintox_bilstm", "main")

@task(name="pwc_clintox_bilstm_hidden_score", benchmark="deltamlbench", family="pwc", variant="hidden_score", provider="anthropic", gpu=True)
def pwc_clintox_bilstm_hidden_score() -> Task:
    return _build_named_task("pwc_clintox_bilstm", "hidden_score")

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

@task(name="pwc_etth1_336_multivariate_softs_main", benchmark="deltamlbench", family="pwc", variant="main", provider="anthropic", gpu=True)
def pwc_etth1_336_multivariate_softs_main() -> Task:
    return _build_named_task("pwc_etth1_336_multivariate_softs", "main")

@task(name="pwc_etth1_336_multivariate_softs_hidden_score", benchmark="deltamlbench", family="pwc", variant="hidden_score", provider="anthropic", gpu=True)
def pwc_etth1_336_multivariate_softs_hidden_score() -> Task:
    return _build_named_task("pwc_etth1_336_multivariate_softs", "hidden_score")

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

@task(name="pwc_kvasir_seg_effisegnet_b5_main", benchmark="deltamlbench", family="pwc", variant="main", provider="anthropic", gpu=True)
def pwc_kvasir_seg_effisegnet_b5_main() -> Task:
    return _build_named_task("pwc_kvasir_seg_effisegnet_b5", "main")

@task(name="pwc_kvasir_seg_effisegnet_b5_hidden_score", benchmark="deltamlbench", family="pwc", variant="hidden_score", provider="anthropic", gpu=True)
def pwc_kvasir_seg_effisegnet_b5_hidden_score() -> Task:
    return _build_named_task("pwc_kvasir_seg_effisegnet_b5", "hidden_score")

@task(name="pwc_kvasir_seg_emcad_main", benchmark="deltamlbench", family="pwc", variant="main", provider="anthropic", gpu=True)
def pwc_kvasir_seg_emcad_main() -> Task:
    return _build_named_task("pwc_kvasir_seg_emcad", "main")

@task(name="pwc_kvasir_seg_emcad_hidden_score", benchmark="deltamlbench", family="pwc", variant="hidden_score", provider="anthropic", gpu=True)
def pwc_kvasir_seg_emcad_hidden_score() -> Task:
    return _build_named_task("pwc_kvasir_seg_emcad", "hidden_score")

@task(name="pwc_kvasir_seg_yolo_sam_2_main", benchmark="deltamlbench", family="pwc", variant="main", provider="anthropic", gpu=True)
def pwc_kvasir_seg_yolo_sam_2_main() -> Task:
    return _build_named_task("pwc_kvasir_seg_yolo_sam_2", "main")

@task(name="pwc_kvasir_seg_yolo_sam_2_hidden_score", benchmark="deltamlbench", family="pwc", variant="hidden_score", provider="anthropic", gpu=True)
def pwc_kvasir_seg_yolo_sam_2_hidden_score() -> Task:
    return _build_named_task("pwc_kvasir_seg_yolo_sam_2", "hidden_score")

@task(name="pwc_malnet_tiny_gatedgcn_main", benchmark="deltamlbench", family="pwc", variant="main", provider="anthropic", gpu=True)
def pwc_malnet_tiny_gatedgcn_main() -> Task:
    return _build_named_task("pwc_malnet_tiny_gatedgcn", "main")

@task(name="pwc_malnet_tiny_gatedgcn_hidden_score", benchmark="deltamlbench", family="pwc", variant="hidden_score", provider="anthropic", gpu=True)
def pwc_malnet_tiny_gatedgcn_hidden_score() -> Task:
    return _build_named_task("pwc_malnet_tiny_gatedgcn", "hidden_score")

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

@task(name="pwc_mnist_gatedgcn_main", benchmark="deltamlbench", family="pwc", variant="main", provider="anthropic", gpu=True)
def pwc_mnist_gatedgcn_main() -> Task:
    return _build_named_task("pwc_mnist_gatedgcn", "main")

@task(name="pwc_mnist_gatedgcn_hidden_score", benchmark="deltamlbench", family="pwc", variant="hidden_score", provider="anthropic", gpu=True)
def pwc_mnist_gatedgcn_hidden_score() -> Task:
    return _build_named_task("pwc_mnist_gatedgcn", "hidden_score")

@task(name="pwc_mnist_rkan_main", benchmark="deltamlbench", family="pwc", variant="main", provider="anthropic", gpu=True)
def pwc_mnist_rkan_main() -> Task:
    return _build_named_task("pwc_mnist_rkan", "main")

@task(name="pwc_mnist_rkan_hidden_score", benchmark="deltamlbench", family="pwc", variant="hidden_score", provider="anthropic", gpu=True)
def pwc_mnist_rkan_hidden_score() -> Task:
    return _build_named_task("pwc_mnist_rkan", "hidden_score")

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

@task(name="pwc_ogbl_ddi_gcn_node_embedding_main", benchmark="deltamlbench", family="pwc", variant="main", provider="anthropic", gpu=True)
def pwc_ogbl_ddi_gcn_node_embedding_main() -> Task:
    return _build_named_task("pwc_ogbl_ddi_gcn_node_embedding", "main")

@task(name="pwc_ogbl_ddi_gcn_node_embedding_hidden_score", benchmark="deltamlbench", family="pwc", variant="hidden_score", provider="anthropic", gpu=True)
def pwc_ogbl_ddi_gcn_node_embedding_hidden_score() -> Task:
    return _build_named_task("pwc_ogbl_ddi_gcn_node_embedding", "hidden_score")

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

@task(name="pwc_stl_10_40_labels_semioccam_main", benchmark="deltamlbench", family="pwc", variant="main", provider="anthropic", gpu=True)
def pwc_stl_10_40_labels_semioccam_main() -> Task:
    return _build_named_task("pwc_stl_10_40_labels_semioccam", "main")

@task(name="pwc_stl_10_40_labels_semioccam_hidden_score", benchmark="deltamlbench", family="pwc", variant="hidden_score", provider="anthropic", gpu=True)
def pwc_stl_10_40_labels_semioccam_hidden_score() -> Task:
    return _build_named_task("pwc_stl_10_40_labels_semioccam", "hidden_score")

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

@task(name="pwc_training_and_validation_dataset_of_capsule_vision_2024_challenge_biomedclip_pubmedbert_main", benchmark="deltamlbench", family="pwc", variant="main", provider="anthropic", gpu=True)
def pwc_training_and_validation_dataset_of_capsule_vision_2024_challenge_biomedclip_pubmedbert_main() -> Task:
    return _build_named_task("pwc_training_and_validation_dataset_of_capsule_vision_2024_challenge_biomedclip_pubmedbert", "main")

@task(name="pwc_training_and_validation_dataset_of_capsule_vision_2024_challenge_biomedclip_pubmedbert_hidden_score", benchmark="deltamlbench", family="pwc", variant="hidden_score", provider="anthropic", gpu=True)
def pwc_training_and_validation_dataset_of_capsule_vision_2024_challenge_biomedclip_pubmedbert_hidden_score() -> Task:
    return _build_named_task("pwc_training_and_validation_dataset_of_capsule_vision_2024_challenge_biomedclip_pubmedbert", "hidden_score")

@task(name="pwc_txl_pbc_a_freely_accessible_labeled_peripheral_blood_cell_dataset_yolov5n_main", benchmark="deltamlbench", family="pwc", variant="main", provider="anthropic", gpu=True)
def pwc_txl_pbc_a_freely_accessible_labeled_peripheral_blood_cell_dataset_yolov5n_main() -> Task:
    return _build_named_task("pwc_txl_pbc_a_freely_accessible_labeled_peripheral_blood_cell_dataset_yolov5n", "main")

@task(name="pwc_txl_pbc_a_freely_accessible_labeled_peripheral_blood_cell_dataset_yolov5n_hidden_score", benchmark="deltamlbench", family="pwc", variant="hidden_score", provider="anthropic", gpu=True)
def pwc_txl_pbc_a_freely_accessible_labeled_peripheral_blood_cell_dataset_yolov5n_hidden_score() -> Task:
    return _build_named_task("pwc_txl_pbc_a_freely_accessible_labeled_peripheral_blood_cell_dataset_yolov5n", "hidden_score")

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
