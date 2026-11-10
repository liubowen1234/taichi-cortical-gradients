# Cortical gradients in Tai Chi practitioners, basketball players and non-athletes: analysis code

This folder contains the code for the main analyses of the paper: functional connectivity (FC) after XCP-D, connectivity
gradients and their alignment, three-group statistics, network segregation and NBS, spatial correspondence with
published cortical maps, training dose, robustness analyses, and main Figs. 2–6.

The code starts from XCP-D outputs. Preprocessing itself (fMRIPrep 25.2.5, XCP-D 26.1.1 with the 36P model and, for one
robustness analysis, the 32P model without global signal regression) is not part of this folder.

## Folder layout

```
config/            config_template.json -> copy to config.json and edit the paths
common/            shared Python code (paths, loaders, statistics engine, gradient helpers, spin test, surface plotting)
matlab/            MATLAB steps: gradients (BrainSpace) and NBS
pipeline/          the analysis steps, numbered in the order they are run (01 ... 07)
resources/         small tables that the code needs: parcel order and network labels of the atlases, and the list of
                   tests in each family used for FDR correction (no participant data)
outputs/           created by the pipeline (location set in config.json)
```

## Requirements

- Python 3.11+ with the packages in `requirements.txt`
- MATLAB (R2024b was used) with [BrainSpace](https://github.com/MICA-MNI/BrainSpace) 0.2.1 and
  [NBS](https://www.nitrc.org/projects/nbs) 1.2; the Statistics Toolbox is used by NBS (`tcdf`, `finv`)
- [Cole-Anticevic Brain-wide Network Partition](https://github.com/ColeLab/ColeAnticevicNetPartition) (CAB-NP) files:
  the Glasser parcels on fs_LR 32k (`SeparateHemispheres/*GlasserParcels_{L,R}.label.gii`) and the whole-brain
  dlabel file and label key (`CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR.*`)
- The BrainSpace data folder (`matlab/datasets/surfaces` for the conte69 surfaces and spheres,
  `matlab/datasets/template_gradients` for the external template)
- For the reference maps only (step 04, s02): neuromaps 0.0.7, NiMARE 0.21.0 and Connectome Workbench (`wb_command`)

## Inputs

1. **XCP-D outputs** for every participant (`<sub>/func/`): `*_motion.tsv`, `*_outliers.tsv`, `*_design.tsv`,
   the unsmoothed `*_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii`, and, where available, the parcellated
   connectivity (`*_atlas-Glasser_*_pearsoncorrelation_boldmap.pconn.nii`, same for 4S456 / 4S656).
   List each XCP-D folder in `config.json` with `fc_source` = `pconn` (use the XCP-D connectivity file) or `dtseries`
   (compute the connectivity from the denoised time series; same result to < 1e-6).
2. **Participant table** (`participants_table` in `config.json`), one row per scanned participant:

   | column | content |
   |---|---|
   | participant_id | e.g. `sub-001` (as in the XCP-D folders) |
   | group | `HC` (non-athletes), `TC` (Tai Chi), `BB` (basketball) |
   | age | years |
   | sex | 1 = female, 2 = male |
   | education_years | years of education |
   | eTIV_mL | estimated total intracranial volume (FreeSurfer) |
   | train_years | years of training (athletes; empty for HC) |
   | weekly_hours, level | optional (basketball) |

## How to run

Copy `config/config_template.json` to `config/config.json` and fill in the paths. Then run the steps below from this
folder (`python pipeline/...`), in this order. Each script explains at the top what it computes and what it writes.

| # | step | script | main outputs |
|---|---|---|---|
| 1.1 | head motion, exclusion (mean FD > 0.25 mm), subject order | `pipeline/01_cohort/s01_motion_and_participants.py` | `01_cohort/subject_covariates.csv` |
| 1.2 | FC matrices (Glasser 36P and 32P; Schaefer-400/600 36P) | `pipeline/01_cohort/s02_build_fc.py --atlas glasser --pipeline 36P` (repeat for `--pipeline 32P`, `--atlas schaefer400`, `--atlas schaefer600`) | `01_cohort/fc/FC_*.mat` |
| 1.3 | participant characteristics | `pipeline/01_cohort/s03_participant_characteristics.py` | `01_cohort/demographics.csv` |
| 2.0 | gradients, reference, alignment, metrics (MATLAB) | `matlab/gradients/run_all_gradients.m` | `02_gradients/glasser358_HCreference/Step1-4_*` |
| 2.1 | eigenvalues, G3 orientation, extra metrics | `pipeline/02_gradients/s01_extra_subject_metrics.py` (and `... noGSR`) | `step_9.1/`, `step_9.2/` |
| 2.2 | three-group statistics of all gradient metrics | `pipeline/02_gradients/s02_group_statistics.py` (optionally `--covar eTIV`) | `stats_three_group/*_3group.csv` |
| 2.3 | alignment to the external template | `pipeline/02_gradients/s03_external_template_alignment.py` | `02_gradients/glasser358_external_template/` |
| 2.4 | similarity of the two athlete effect maps (label permutation) | `pipeline/02_gradients/s04_effect_map_similarity.py` | `effect_map_similarity_labelperm.csv` |
| 3.1 | network segregation, participation coefficient | `pipeline/03_network_connectivity/s01_network_segregation.py` | `03_network_connectivity/segregation/` |
| 3.2 | VIS2 connectivity profile; segregation vs G2 | `pipeline/03_network_connectivity/s02_visual2_connectivity.py` | `03_network_connectivity/visual2_connectivity/` |
| 3.3 | VIS2 partner contrast (SMN vs CON) | `pipeline/03_network_connectivity/s03_visual2_partner_contrast.py` | same folder |
| 3.4 | NBS (MATLAB), then summary | `matlab/nbs/run_nbs_three_group.m`, then `pipeline/03_network_connectivity/s04_nbs_summary.py` | `03_network_connectivity/nbs/` |
| 4.1 | spin permutations | `pipeline/04_spatial/s01_make_spin_permutations.py` | `04_spatial/spin_permutations/` |
| 4.2 | download and project reference maps | `pipeline/04_spatial/s02_fetch_reference_maps.py` | `04_spatial/map_contextualization/data/` |
| 4.3 | spatial correspondence with reference maps | `pipeline/04_spatial/s03_map_contextualization.py` | `spatial_correspondence_3group.csv` |
| 4.4–4.5 | subcortical / cerebellar projection | `pipeline/04_spatial/s04_extract_cabnp_timeseries.py`, `s05_subcortical_projection.py` | `04_spatial/subcortical_projection/` |
| 5 | training dose | `pipeline/05_training_dose/s01_training_dose.py` | `05_training_dose/` |
| 6.1 | head-motion subsets | `pipeline/06_robustness/s01_motion_subsets.py` | `06_robustness/motion_subsets/` |
| 6.2–6.4 | signal quality, QC-FC, motion maps | `s02_signal_quality.py`, `s03_motion_qc_and_quality_stats.py`, `s04_motion_network_enrichment.py` | `06_robustness/data_quality/` |
| 6.5 | global signal amplitude | `s05_global_signal_amplitude.py` | `06_robustness/global_signal_amplitude/` |
| 6.6 | no global signal regression (32P) | `s06_no_GSR_comparison.py` | `06_robustness/no_GSR_32P/` |
| 6.7 | Schaefer-400 / 600 | `s07_schaefer_statistics.py 400`, `s07_schaefer_statistics.py 600` | `06_robustness/schaefer*/stats_three_group/` |
| 6.8 | specification curve (1,000 specifications; slow) | `s08_specification_curve.py` | `06_robustness/multiverse/` |
| 6.9 | alignment comparison, frame rotation | `s09_alignment_comparison.py` | `06_robustness/alignment_comparison/` |
| 6.10 | uniform robustness summary, Fig. 6 data | `s10_robustness_summary.py` | `06_robustness/robustness_summary/` |
| 7 | main figures 2–6 | `pipeline/07_figures/fig2_three_gradients.py` ... `fig6_robustness.py` | `07_figures/` |

`run_pipeline.sh` runs the Python steps in this order; the two MATLAB steps are run separately (it stops and tells you
when).

## Main analysis choices (see the Methods of the paper)

- **Parcels**: HCP-MMP (Glasser) 360 parcels in the XCP-D order (right hemisphere first); the two hippocampal parcels
  are removed (almost no signal), leaving 358. Networks: CAB-NP, 12 cortical networks. Yeo-7 labels are used only to
  orient G1 and G2.
- **Gradients** (BrainSpace): diagonal set to 0, each row keeps its strongest 10% of connections, cosine affinity,
  diffusion maps (alpha = 0.5, diffusion time 0), 10 components.
- **Reference and alignment**: the non-athlete FC matrices are averaged (Fisher z) and embedded with the same settings;
  each subject's first three components are rotated onto the reference with one orthogonal Procrustes rotation.
  G1 = default mode vs sensory, G2 = visual vs somatosensory (visual positive), G3 = remaining component, signed so that
  the modulation networks are positive.
- **Metrics**: span = 95th − 5th percentile of the parcel scores; eigenvalues from each subject's own embedding;
  G1–G2 plane: dispersion, major- and minor-axis variance, anisotropy.
- **Statistics** (`common/stats_engine.py`): `Metric ~ TC + BB + Age + Sex + MeanFD`, OLS with HC3 standard errors;
  omnibus 2-df Wald F with BH-FDR within each family (`resources/test_families/`), then pairwise contrasts with Holm
  correction; Hedges' g, TOST equivalence (±0.5 SD) and JZS Bayes factors.
- **Spatial tests**: spin test (10,000 rotations, Hungarian assignment); correlations with and without the reference
  G2 axis regressed out.

## Notes

- Every subject-level table has the same row order (HC, then TC, then BB; ascending ID within group). The MATLAB steps
  label subjects by row number (`sub-001`, `sub-002`, ...).
- Fixed random seeds are set where permutations are drawn, so the results are reproducible.
- Fig. 1 (study design schematic) is a drawing and is not produced by this code.
