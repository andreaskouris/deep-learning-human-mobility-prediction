# PHASE6 exact public selection

This candidate is the intended single thesis repository. PHASE5 is preserved separately. Each path, role, required stage/experiment, evidence and SHA256 is in the [machine-readable inventory](../reproducibility/file_selection.json). CONFIRMED describes current verified dependency/source use, not a claim that new integration files executed in historical training.

## INCLUDED authored code and publication material

| Path | Class | Role |
| --- | --- | --- |
| `.gitattributes` | DOCUMENTATION | Pin source LF bytes and preserve archived result/figure bytes across Git checkouts |
| `.gitignore` | DOCUMENTATION | Publication/reproducibility documentation and declarative artifact/source contract |
| `assets/figures/figure_5_1_next_hop_aggregate_performance.svg` | FIGURE | Archived final aggregate figure displayed by README |
| `assets/figures/figure_5_2_next_hop_paired_bootstrap_ci.svg` | FIGURE | Archived final aggregate figure displayed by README |
| `assets/figures/figure_5_5_tte_aggregate_performance.svg` | FIGURE | Archived final aggregate figure displayed by README |
| `assets/figures/figure_5_6_tte_paired_bootstrap_ci.svg` | FIGURE | Archived final aggregate figure displayed by README |
| `CITATION.md` | DOCUMENTATION | Publication/reproducibility documentation and declarative artifact/source contract |
| `config/__init__.py` | ORIGINAL_REVIEWED | Independent candidate namespace marker; generated upstream initializer replaces it only in local runtime |
| `config/thesis_paths.py` | ORIGINAL_REVIEWED | Bind the canonical split/seed/semantic/cache paths; 120 required values compared across six experiments |
| `data_provider/__init__.py` | ORIGINAL_REVIEWED | Package marker; independently authored for data_provider/utils |
| `data_provider/cached_next_hop_dataset.py` | ORIGINAL_REVIEWED | Read and validate prebuilt NH trajectory tensors |
| `data_provider/dynamic_preprocessing.py` | ORIGINAL_REVIEWED | Canonical controlled observed-zero preserving traffic preprocessing |
| `data_provider/tte_dataset.py` | ORIGINAL_REVIEWED | Build/validate departure-only TTE payloads and read cached splits |
| `docs/comment_cleanup.md` | DOCUMENTATION | Publication/reproducibility documentation and declarative artifact/source contract |
| `docs/file_selection.md` | DOCUMENTATION | Publication/reproducibility documentation and declarative artifact/source contract |
| `docs/installation.md` | DOCUMENTATION | Actual source acquisition/integration/verification CLI and separate observed environments |
| `docs/reproducibility.md` | DOCUMENTATION | Publication/reproducibility documentation and declarative artifact/source contract |
| `docs/reproduction.md` | DOCUMENTATION | Complete canonical preprocessing, external assets, fresh/historical training/evaluation workflow |
| `docs/UPSTREAM_CHANGES.md` | DOCUMENTATION | Original seven plus eighth portable adaptation, generated-source ownership and reused-body disposition |
| `docs/verification.md` | DOCUMENTATION | Publication/reproducibility documentation and declarative artifact/source contract |
| `docs/workflow.md` | DOCUMENTATION | Publication/reproducibility documentation and declarative artifact/source contract |
| `environments/analysis.txt` | DOCUMENTATION | Observed version snapshot/provenance for a separate environment |
| `environments/bigcity.txt` | DOCUMENTATION | Observed version snapshot/provenance for a separate environment |
| `environments/provenance.json` | DOCUMENTATION | Observed version snapshot/provenance for a separate environment |
| `environments/qwen.txt` | DOCUMENTATION | Observed version snapshot/provenance for a separate environment |
| `experiments/nh_baseline.json` | DOCUMENTATION | Canonical experiment specification with per-field provenance |
| `experiments/nh_poi_tfidf.json` | DOCUMENTATION | Canonical experiment specification with per-field provenance |
| `experiments/nh_qwen_poi.json` | DOCUMENTATION | Canonical experiment specification with per-field provenance |
| `experiments/tte_baseline.json` | DOCUMENTATION | Canonical experiment specification with per-field provenance |
| `experiments/tte_qwen.json` | DOCUMENTATION | Canonical experiment specification with per-field provenance |
| `experiments/tte_structured.json` | DOCUMENTATION | Canonical experiment specification with per-field provenance |
| `LICENSE_STATUS.md` | DOCUMENTATION | Publication/reproducibility documentation and declarative artifact/source contract |
| `models/__init__.py` | ORIGINAL_REVIEWED | Independent namespace marker for published task extensions |
| `models/bigcity4tte.py` | ORIGINAL_REVIEWED | Original TTE head, masked attention, semantic gating, REG packing and duration/loss path |
| `models/thesis_semantics.py` | ORIGINAL_REVIEWED | Original semantic normalization/zero projection and pretrained state_dict compatibility |
| `models/tte_tokenization_adapter.py` | ORIGINAL_REVIEWED | Independent attribute-view attention hook; reuse corrected upstream token operations with canonical TTE PAD mask |
| `README.md` | DOCUMENTATION | Publication/reproducibility documentation and declarative artifact/source contract |
| `reproducibility/dependency_chains.json` | DOCUMENTATION | Revised source integration and upstream-change provenance contract |
| `reproducibility/external_artifacts.json` | DOCUMENTATION | Publication/reproducibility documentation and declarative artifact/source contract |
| `reproducibility/integration_inputs.json` | DOCUMENTATION | Pinned 13 upstream preimages/eight derived fingerprints plus 73 authored runtime/support source hashes |
| `reproducibility/runner_interface.json` | DOCUMENTATION | Publication/reproducibility documentation and declarative artifact/source contract |
| `reproducibility/source_equivalence.json` | DOCUMENTATION | Sixteen independent canonical source/AST comparisons, with explicit specialization and real-weight limits |
| `reproducibility/standalone_runtime_plan.json` | DOCUMENTATION | Publication/reproducibility documentation and declarative artifact/source contract |
| `results/canonical_metrics.json` | AGGREGATE_RESULT | Archived aggregate results; no per-sample trajectories/predictions |
| `results/next_hop/next_hop_model_metrics.csv` | AGGREGATE_RESULT | Archived aggregate results; no per-sample trajectories/predictions |
| `results/next_hop/next_hop_pairwise_statistics.csv` | AGGREGATE_RESULT | Archived aggregate results; no per-sample trajectories/predictions |
| `results/tte/tte_model_error_statistics.csv` | AGGREGATE_RESULT | Archived aggregate results; no per-sample trajectories/predictions |
| `results/tte/tte_pairwise_statistics.csv` | AGGREGATE_RESULT | Archived aggregate results; no per-sample trajectories/predictions |
| `scripts/__init__.py` | ORIGINAL_REVIEWED | Package marker; independently authored for data_provider/utils |
| `scripts/analyze_final_next_hop_statistics.py` | ORIGINAL_REVIEWED | Paired NH aggregate metrics, McNemar/Holm tests and bootstrap |
| `scripts/analyze_final_next_hop_stratified.py` | ORIGINAL_REVIEWED | Align NH outputs and stratify by canonical task/context bins |
| `scripts/analyze_final_tte_statistics.py` | ORIGINAL_REVIEWED | Paired TTE absolute/relative errors, sign/Holm tests and bootstrap |
| `scripts/analyze_final_tte_stratified.py` | ORIGINAL_REVIEWED | Compare TTE errors by fixed bins and paired strata |
| `scripts/build_dynamic_cache.py` | ORIGINAL_REVIEWED | Independent chunked .dyna decoder and controlled cache writer/reloader |
| `scripts/build_full_metadata.py` | ORIGINAL_REVIEWED | Compute five full-dataset counts via read-only FileLoader functions |
| `scripts/build_language_road_semantic_cache.py` | ORIGINAL_REVIEWED | Combine TYPE prototypes using canonical road TF-IDF weights |
| `scripts/build_language_semantic_cache.py` | ORIGINAL_REVIEWED | Create normalized Qwen TYPE prototypes from unique POI names |
| `scripts/build_next_hop_cache.py` | ORIGINAL_REVIEWED | Build/validate NH trajectory splits using DatasetNextHop |
| `scripts/build_semantic_cache.py` | ORIGINAL_REVIEWED | Build 204-feature POI TYPE count/IDF road features |
| `scripts/build_structural_caches.py` | ORIGINAL_REVIEWED | Build/validate shared relation and static caches via FileLoader |
| `scripts/build_tte_qwen_semantic_cache.py` | ORIGINAL_REVIEWED | Embed the same ten facts and normalize road concept sums |
| `scripts/build_tte_structured_semantic_cache.py` | ORIGINAL_REVIEWED | Rebuild/verify ten binary facts from road/POI/link source files |
| `scripts/export_final_next_hop_per_sample.py` | ORIGINAL_REVIEWED | Export selected NH checkpoints in historical/fresh profiles |
| `scripts/export_final_tte_per_sample.py` | ORIGINAL_REVIEWED | Export selected TTE checkpoints in historical/fresh profiles |
| `scripts/plot_final_next_hop_figures_5_1_5_2.py` | ORIGINAL_REVIEWED | Plot canonical NH aggregate metrics and paired intervals |
| `scripts/plot_final_next_hop_figures_5_3_5_4.py` | ORIGINAL_REVIEWED | Plot canonical NH stratified results |
| `scripts/plot_final_tte_figures.py` | ORIGINAL_REVIEWED | Plot canonical TTE aggregate and stratified comparisons |
| `scripts/portable_runtime_support.py` | ORIGINAL_REVIEWED | Resolve portable execution arguments; preserve historical/fresh integrity policy |
| `scripts/run_selected_tests.py` | VERIFICATION | Guarded tooling/science/integrated-runtime groups; optional separate historical reference group |
| `scripts/validate_experiment_configs.py` | VERIFICATION | AST-read the locally generated parser/runners and validate canonical artifact/environment contracts |
| `scripts/verify_public_candidate.py` | VERIFICATION | Exact public inventory, source interfaces, external/generated import classification and figures validation |
| `scripts/verify_tte_dataset.py` | ORIGINAL_REVIEWED | Build/verify canonical TTE split caches from source trajectories |
| `setup/__init__.py` | ORIGINAL_REVIEWED | Independent integration control package marker |
| `setup/integrate_bigcity.py` | ORIGINAL_REVIEWED | Standalone pinned-checkout validation, dry-run, eight source backups, transactional integration and read-only repeat verification |
| `setup/source_transformations.py` | ORIGINAL_REVIEWED | Eight transparent AST edits to user-provided upstream nodes; no full replacement source or encoded payload |
| `tests/test_build_dynamic_cache.py` | VERIFICATION | Selected canonical synthetic regression/verification contract: test_build_dynamic_cache.py |
| `tests/test_build_full_metadata.py` | VERIFICATION | Selected canonical synthetic regression/verification contract: test_build_full_metadata.py |
| `tests/test_build_next_hop_cache.py` | VERIFICATION | Selected canonical synthetic regression/verification contract: test_build_next_hop_cache.py |
| `tests/test_build_structural_caches.py` | VERIFICATION | Selected canonical synthetic regression/verification contract: test_build_structural_caches.py |
| `tests/test_build_tte_structured_semantic_cache.py` | VERIFICATION | Selected canonical synthetic regression/verification contract: test_build_tte_structured_semantic_cache.py |
| `tests/test_cached_next_hop_dataset.py` | VERIFICATION | Selected canonical synthetic regression/verification contract: test_cached_next_hop_dataset.py |
| `tests/test_canonical_runtime_regressions.py` | VERIFICATION | Selected canonical synthetic regression/verification contract: test_canonical_runtime_regressions.py |
| `tests/test_dynamic_preprocessing.py` | VERIFICATION | Selected canonical synthetic regression/verification contract: test_dynamic_preprocessing.py |
| `tests/test_experiment_configs.py` | VERIFICATION | Selected canonical synthetic regression/verification contract: test_experiment_configs.py |
| `tests/test_file_loader_dynamic_modes.py` | VERIFICATION | Selected canonical synthetic regression/verification contract: test_file_loader_dynamic_modes.py |
| `tests/test_integrate_bigcity.py` | ORIGINAL_REVIEWED | Fifteen synthetic integration safety/transaction/preimage/idempotence tests |
| `tests/test_next_hop_metrics.py` | VERIFICATION | Selected canonical synthetic regression/verification contract: test_next_hop_metrics.py |
| `tests/test_portable_runtime_interfaces.py` | VERIFICATION | Selected canonical synthetic regression/verification contract: test_portable_runtime_interfaces.py |
| `tests/test_portable_runtime_support.py` | VERIFICATION | Selected canonical synthetic regression/verification contract: test_portable_runtime_support.py |
| `tests/test_public_candidate.py` | VERIFICATION | Selected canonical synthetic regression/verification contract: test_public_candidate.py |
| `tests/test_revised_implementation.py` | VERIFICATION | Synthetic relocation/initialization/state_dict/checkpoint/controlled-equivalence tests |
| `tests/test_train_next_hop.py` | VERIFICATION | Selected canonical synthetic regression/verification contract: test_train_next_hop.py |
| `tests/test_tte_metrics.py` | VERIFICATION | Selected canonical synthetic regression/verification contract: test_tte_metrics.py |
| `tests/test_tte_tokenizer_hook.py` | ORIGINAL_REVIEWED | Two optional reference-only token/gradient comparisons using tiny CPU fixtures |
| `THIRD_PARTY_NOTICES.md` | DOCUMENTATION | Publication/reproducibility documentation and declarative artifact/source contract |
| `train_next_hop.py` | ORIGINAL_REVIEWED | Canonical NH training/strict validation selection/test/export orchestration |
| `train_tte.py` | ORIGINAL_REVIEWED | Canonical TTE training, train-only normalization, MAE selection and seconds-based evaluation |
| `utils/__init__.py` | ORIGINAL_REVIEWED | Package marker; independently authored for data_provider/utils |
| `utils/next_hop_metrics.py` | ORIGINAL_REVIEWED | Compute NH accuracy, MRR@5 and NDCG@5 |
| `utils/tte_metrics.py` | ORIGINAL_REVIEWED | Accumulate unclipped TTE MAE/RMSE/MAPE |
| `reproducibility/file_selection.json` | DOCUMENTATION | Self-describing exact included/excluded inventory and external dependency evidence |

## External official source inputs

Eight original paths are transformed locally; five remain byte-identical. No protected replacement body is published. The [input specification](../reproducibility/integration_inputs.json) pins commit/tree, all thirteen preimages, eight output fingerprints and authored runtime source hashes.

| Path | Handling |
| --- | --- |
| `config/__init__.py` | LOCALLY_TRANSFORMED_UPSTREAM |
| `config/args_config.py` | LOCALLY_TRANSFORMED_UPSTREAM |
| `config/global_vars.py` | LOCALLY_TRANSFORMED_UPSTREAM |
| `config/logging_config.py` | OFFICIAL_UPSTREAM |
| `config/random_seed.py` | OFFICIAL_UPSTREAM |
| `data_provider/data_loader.py` | LOCALLY_TRANSFORMED_UPSTREAM |
| `data_provider/file_loader.py` | LOCALLY_TRANSFORMED_UPSTREAM |
| `models/backbone.py` | LOCALLY_TRANSFORMED_UPSTREAM |
| `models/bigcity.py` | OFFICIAL_UPSTREAM |
| `models/bigcity4finetune.py` | LOCALLY_TRANSFORMED_UPSTREAM |
| `models/layers.py` | OFFICIAL_UPSTREAM |
| `models/st_tokenizer.py` | LOCALLY_TRANSFORMED_UPSTREAM |
| `utils/timefeatures.py` | OFFICIAL_UPSTREAM |

The legacy manifest field name `external_private_runtime_only` is retained for schema compatibility: PHASE6 has no private assembly provider. All thirteen entries are supplied by the explicit official checkout. The independent TTE hook is public; legacy original mode is excluded.

## EXCLUDED tracked research source/material

| Path | Classification |
| --- | --- |
| `.gitignore` | EXCLUDED |
| `README.md` | EXCLUDED |
| `analysis/external_ghsl/ghsl_built_surface_2018_R5_C28_verification.json` | EXCLUDED |
| `analysis/external_ghsl/ghsl_nres_2018_R5_C28_verification.json` | EXCLUDED |
| `analysis/external_osm/ohsome_historical_coverage_audit.csv` | EXCLUDED |
| `analysis/external_osm/ohsome_historical_coverage_audit.json` | EXCLUDED |
| `analysis/per_sample/results/analysis_report.txt` | EXCLUDED |
| `analysis/per_sample/results/analysis_summary.json` | EXCLUDED |
| `analysis/per_sample/results/correctness_pattern_summary.csv` | EXCLUDED |
| `analysis/per_sample/results/model_summary.csv` | EXCLUDED |
| `analysis/per_sample/results/output_manifest.json` | EXCLUDED |
| `analysis/per_sample/results/pairwise_by_context_length_bin.csv` | EXCLUDED |
| `analysis/per_sample/results/pairwise_by_last_road_semantics.csv` | EXCLUDED |
| `analysis/per_sample/results/pairwise_by_semantic_coverage.csv` | EXCLUDED |
| `analysis/per_sample/results/pairwise_summary.csv` | EXCLUDED |
| `analysis/per_sample/results/performance_by_any_context_semantics.csv` | EXCLUDED |
| `analysis/per_sample/results/performance_by_context_length_bin.csv` | EXCLUDED |
| `analysis/per_sample/results/performance_by_context_length_exact.csv` | EXCLUDED |
| `analysis/per_sample/results/performance_by_last_road_semantics.csv` | EXCLUDED |
| `analysis/per_sample/results/performance_by_semantic_coverage.csv` | EXCLUDED |
| `analysis/per_sample/results/top1_agreement_matrix.csv` | EXCLUDED |
| `analysis/tte_per_sample/results/tte_exp0_vs_exp1_by_concept.csv` | EXCLUDED |
| `analysis/tte_per_sample/results/tte_exp0_vs_exp1_by_duration.csv` | EXCLUDED |
| `analysis/tte_per_sample/results/tte_exp0_vs_exp1_by_semantic_exposure.csv` | EXCLUDED |
| `analysis/tte_per_sample/results/tte_exp0_vs_exp1_manifest.json` | EXCLUDED |
| `analysis/tte_per_sample/results/tte_exp0_vs_exp1_per_sample.csv` | EXCLUDED |
| `analysis/tte_per_sample/results/tte_exp0_vs_exp1_summary.json` | EXCLUDED |
| `config/__init__.py` | ADAPTATION |
| `config/args_config.py` | ADAPTATION |
| `config/global_vars.py` | ADAPTATION |
| `config/logging_config.py` | UPSTREAM |
| `config/random_seed.py` | UPSTREAM |
| `data_provider/__init__.py` | UPSTREAM |
| `data_provider/data_loader.py` | ADAPTATION |
| `data_provider/file_loader.py` | ADAPTATION |
| `data_provider/m4.py` | EXCLUDED |
| `data_provider/uea.py` | EXCLUDED |
| `download.py` | EXCLUDED |
| `evaluate.py` | EXCLUDED |
| `final_analysis/next_hop/figures/figure_5_1_next_hop_aggregate_performance.pdf` | EXCLUDED |
| `final_analysis/next_hop/figures/figure_5_2_next_hop_paired_bootstrap_ci.pdf` | EXCLUDED |
| `final_analysis/next_hop/figures/figure_5_3_next_hop_accuracy_gain_semantic_coverage.pdf` | EXCLUDED |
| `final_analysis/next_hop/figures/figure_5_3_next_hop_accuracy_gain_semantic_coverage.svg` | EXCLUDED |
| `final_analysis/next_hop/figures/figure_5_4_next_hop_accuracy_gain_context_length.pdf` | EXCLUDED |
| `final_analysis/next_hop/figures/figure_5_4_next_hop_accuracy_gain_context_length.svg` | EXCLUDED |
| `final_analysis/next_hop/statistics/next_hop_correctness_transitions.csv` | EXCLUDED |
| `final_analysis/next_hop/statistics/next_hop_statistics_summary.json` | EXCLUDED |
| `final_analysis/next_hop/stratified/next_hop_per_sample_stratified.csv` | EXCLUDED |
| `final_analysis/next_hop/stratified/next_hop_strata_support.csv` | EXCLUDED |
| `final_analysis/next_hop/stratified/next_hop_stratification_manifest.json` | EXCLUDED |
| `final_analysis/next_hop/stratified/next_hop_stratified_metrics.csv` | EXCLUDED |
| `final_analysis/next_hop/stratified/next_hop_stratified_pairwise.csv` | EXCLUDED |
| `final_analysis/tte/figures/figure_5_5_tte_aggregate_performance.pdf` | EXCLUDED |
| `final_analysis/tte/figures/figure_5_6_tte_paired_bootstrap_ci.pdf` | EXCLUDED |
| `final_analysis/tte/figures/figure_5_7_tte_mae_improvement_by_duration.pdf` | EXCLUDED |
| `final_analysis/tte/figures/figure_5_7_tte_mae_improvement_by_duration.svg` | EXCLUDED |
| `final_analysis/tte/figures/figure_5_8_tte_mape_improvement_by_duration.pdf` | EXCLUDED |
| `final_analysis/tte/figures/figure_5_8_tte_mape_improvement_by_duration.svg` | EXCLUDED |
| `final_analysis/tte/figures/figure_5_9_tte_absolute_error_ecdf.pdf` | EXCLUDED |
| `final_analysis/tte/figures/figure_5_9_tte_absolute_error_ecdf.svg` | EXCLUDED |
| `final_analysis/tte/statistics/tte_statistics_summary.json` | EXCLUDED |
| `final_analysis/tte/stratified/tte_concept_presence_metrics.csv` | EXCLUDED |
| `final_analysis/tte/stratified/tte_duration_group_bootstrap_ci.csv` | EXCLUDED |
| `final_analysis/tte/stratified/tte_strata_support.csv` | EXCLUDED |
| `final_analysis/tte/stratified/tte_stratification_manifest.json` | EXCLUDED |
| `final_analysis/tte/stratified/tte_stratified_metrics.csv` | EXCLUDED |
| `final_analysis/tte/stratified/tte_stratified_pairwise_improvements.csv` | EXCLUDED |
| `final_results/tte/01_baseline/tte_history.csv` | EXCLUDED |
| `final_results/tte/01_baseline/tte_results.json` | EXCLUDED |
| `final_results/tte/02_structured/tte_history.csv` | EXCLUDED |
| `final_results/tte/02_structured/tte_results.json` | EXCLUDED |
| `final_results/tte/03_qwen/tte_history.csv` | EXCLUDED |
| `final_results/tte/03_qwen/tte_results.json` | EXCLUDED |
| `finetune.py` | EXCLUDED |
| `finetune_ddp.py` | EXCLUDED |
| `image/Article/Figure2.png` | EXCLUDED |
| `image/Article/Figure4.png` | EXCLUDED |
| `image/Article/MTSD_Model.png` | EXCLUDED |
| `image/Article/model_architecture.png` | EXCLUDED |
| `image/Article/template_1.png` | EXCLUDED |
| `image/Article/template_2.png` | EXCLUDED |
| `image/Article/training.png` | EXCLUDED |
| `image/Experiments/exp_figure1.png` | EXCLUDED |
| `image/Experiments/exp_figure2.png` | EXCLUDED |
| `make_dataset_cache.py` | EXCLUDED |
| `models/__init__.py` | UPSTREAM |
| `models/backbone.py` | ADAPTATION |
| `models/bigcity.py` | UPSTREAM |
| `models/bigcity4finetune.py` | ADAPTATION |
| `models/gpt2/config.json` | EXCLUDED |
| `models/layers.py` | UPSTREAM |
| `models/st_tokenizer.py` | ADAPTATION |
| `pretrain.py` | EXCLUDED |
| `pretrain_ddp.py` | EXCLUDED |
| `reproducibility/phase2_configurations.md` | EXCLUDED |
| `reproducibility/phase3_portability_inventory.json` | EXCLUDED |
| `reproducibility/phase3_runtime_assembly.md` | EXCLUDED |
| `reproducibility/phase3_verification.json` | EXCLUDED |
| `reproducibility/phase4_standalone_portability.md` | EXCLUDED |
| `reproducibility/phase4_verification.json` | EXCLUDED |
| `reproducibility/runtime_assembly_plan.json` | EXCLUDED |
| `requirements.txt` | EXCLUDED |
| `scripts/analyze_next_hop_per_sample.py` | EXCLUDED |
| `scripts/analyze_tte_per_sample.py` | EXCLUDED |
| `scripts/assemble_bigcity_runtime.py` | EXCLUDED |
| `scripts/assemble_standalone_runtime.py` | EXCLUDED |
| `scripts/audit_historical_osm_tte_coverage.py` | EXCLUDED |
| `scripts/audit_historical_osm_tte_extended.py` | EXCLUDED |
| `scripts/benchmark_next_hop.py` | EXCLUDED |
| `scripts/benchmark_next_hop_validation.py` | EXCLUDED |
| `scripts/benchmark_pretrain.py` | EXCLUDED |
| `scripts/build_bge_language_road_semantic_cache.py` | EXCLUDED |
| `scripts/build_bge_language_semantic_cache.py` | EXCLUDED |
| `scripts/build_time_reg_cache.py` | EXCLUDED |
| `scripts/build_traffic_state_reg_cache.py` | EXCLUDED |
| `scripts/build_traj_classify_cache.py` | EXCLUDED |
| `scripts/build_traj_recover_cache.py` | EXCLUDED |
| `scripts/export_next_hop_per_sample.py` | EXCLUDED |
| `scripts/smoke_test_next_hop.py` | EXCLUDED |
| `scripts/smoke_test_tte.py` | EXCLUDED |
| `scripts/smoke_test_tte_semantics.py` | EXCLUDED |
| `scripts/smoke_train_tte.py` | EXCLUDED |
| `scripts/validate_tte_qwen_runtime.py` | EXCLUDED |
| `scripts/verify_ghsl_built_surface_2018.py` | EXCLUDED |
| `scripts/verify_ghsl_nres_2018.py` | EXCLUDED |
| `scripts/verify_next_hop_time_ids.py` | EXCLUDED |
| `scripts/verify_runtime_assembly.py` | EXCLUDED |
| `tests/test_build_time_reg_cache.py` | EXCLUDED |
| `tests/test_build_traffic_state_reg_cache.py` | EXCLUDED |
| `tests/test_build_traj_classify_cache.py` | EXCLUDED |
| `tests/test_build_traj_recover_cache.py` | EXCLUDED |
| `tests/test_cache_signatures.py` | EXCLUDED |
| `tests/test_dynamic_mode_config.py` | EXCLUDED |
| `tests/test_finetune_runtime.py` | EXCLUDED |
| `tests/test_prompt_graph_regression.py` | EXCLUDED |
| `tests/test_runtime_assembly.py` | EXCLUDED |
| `tests/test_semantic_enrichment.py` | EXCLUDED |
| `tests/test_standalone_runtime.py` | EXCLUDED |
| `utils/__init__.py` | UPSTREAM |
| `utils/losses.py` | EXCLUDED |
| `utils/m4_summary.py` | EXCLUDED |
| `utils/masking.py` | EXCLUDED |
| `utils/metrics.py` | EXCLUDED |
| `utils/plot_losses.py` | EXCLUDED |
| `utils/round_iterator.py` | EXCLUDED |
| `utils/scheduler.py` | EXCLUDED |
| `utils/short_data.py` | EXCLUDED |
| `utils/timefeatures.py` | UPSTREAM |
| `utils/tools.py` | EXCLUDED |

## EXCLUDED untracked research artifacts

- `final_analysis/next_hop/per_sample/next_hop_per_sample_master.csv`.
- `final_analysis/next_hop/per_sample/nh1_baseline_predictions.csv`.
- `final_analysis/next_hop/per_sample/nh2_poi_tfidf_predictions.csv`.
- `final_analysis/next_hop/per_sample/nh3_qwen_poi_predictions.csv`.
- `final_analysis/tte/per_sample/tte_per_sample.csv`.
- `final_analysis/tte/per_sample/tte_per_sample_summary.json`.
- `final_analysis_script_audit.txt`.
- `final_analysis_source_scripts.zip`.
- `final_results/next_hop/01_baseline/checkpoint/best_next_hop.pth`.
- `final_results/next_hop/02_poi_tfidf/checkpoint/best_next_hop.pth`.
- `final_results/next_hop/03_qwen_poi/checkpoint/best_next_hop.pth`.
- `final_results/tte/01_baseline/checkpoint/best_tte_none_seed2021.pth`.
- `final_results/tte/02_structured/checkpoint/best_tte_tte_structured_seed2021.pth`.
- `final_results/tte/03_qwen/checkpoint/best_tte_tte_qwen_seed2021.pth`.
- `next_hop_logging_logic.txt`.
- `train_tte_before_logging_fix.py`.
- `tte_help.txt`.

All unlisted/generated data, weights, caches, prediction records, private backups and derived upstream output remain excluded. Historical source hashes are metadata for optional comparison, not an installation dependency.
