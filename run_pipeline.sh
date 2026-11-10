#!/bin/bash
# Run the Python steps of the pipeline in order. The two MATLAB steps (gradients, NBS) must be run in MATLAB when the
# script stops and asks for them; then start this script again with the step number shown, e.g.  FROM=2 ./run_pipeline.sh
# Usage (from the code_release folder):  ./run_pipeline.sh          or   FROM=<stage> ./run_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
FROM=${FROM:-1}

if [ "$FROM" -le 1 ]; then
  $PY pipeline/01_cohort/s01_motion_and_participants.py
  $PY pipeline/01_cohort/s02_build_fc.py --atlas glasser --pipeline 36P
  $PY pipeline/01_cohort/s02_build_fc.py --atlas glasser --pipeline 32P
  $PY pipeline/01_cohort/s02_build_fc.py --atlas schaefer400 --pipeline 36P
  $PY pipeline/01_cohort/s02_build_fc.py --atlas schaefer600 --pipeline 36P
  $PY pipeline/01_cohort/s03_participant_characteristics.py
  echo "Now run matlab/gradients/run_all_gradients.m in MATLAB, then:  FROM=2 ./run_pipeline.sh"
  exit 0
fi

if [ "$FROM" -le 2 ]; then
  $PY pipeline/02_gradients/s01_extra_subject_metrics.py main
  $PY pipeline/02_gradients/s01_extra_subject_metrics.py noGSR
  $PY pipeline/02_gradients/s02_group_statistics.py main
  $PY pipeline/02_gradients/s02_group_statistics.py main --covar eTIV
  $PY pipeline/02_gradients/s03_external_template_alignment.py
  $PY pipeline/02_gradients/s03_external_template_alignment.py --covar eTIV
  $PY pipeline/02_gradients/s04_effect_map_similarity.py
  $PY pipeline/03_network_connectivity/s01_network_segregation.py
  $PY pipeline/03_network_connectivity/s01_network_segregation.py --covar eTIV
  $PY pipeline/03_network_connectivity/s02_visual2_connectivity.py
  $PY pipeline/03_network_connectivity/s03_visual2_partner_contrast.py
  echo "Now run matlab/nbs/run_nbs_three_group.m in MATLAB, then:  FROM=3 ./run_pipeline.sh"
  exit 0
fi

$PY pipeline/03_network_connectivity/s04_nbs_summary.py
$PY pipeline/04_spatial/s01_make_spin_permutations.py
$PY pipeline/04_spatial/s02_fetch_reference_maps.py          # needs neuromaps, NiMARE and wb_command
$PY pipeline/04_spatial/s03_map_contextualization.py
$PY pipeline/04_spatial/s04_extract_cabnp_timeseries.py
$PY pipeline/04_spatial/s05_subcortical_projection.py
$PY pipeline/05_training_dose/s01_training_dose.py
$PY pipeline/06_robustness/s01_motion_subsets.py
$PY pipeline/06_robustness/s02_signal_quality.py
$PY pipeline/06_robustness/s03_motion_qc_and_quality_stats.py
$PY pipeline/06_robustness/s04_motion_network_enrichment.py
$PY pipeline/06_robustness/s05_global_signal_amplitude.py
$PY pipeline/06_robustness/s06_no_GSR_comparison.py
$PY pipeline/06_robustness/s07_schaefer_statistics.py 400
$PY pipeline/06_robustness/s07_schaefer_statistics.py 600
$PY pipeline/06_robustness/s08_specification_curve.py          # many hours
$PY pipeline/06_robustness/s09_alignment_comparison.py
$PY pipeline/06_robustness/s10_robustness_summary.py
for f in fig2_three_gradients fig3_G2_location fig4_VIS2_segregation fig5_training_dose fig6_robustness; do
  $PY pipeline/07_figures/$f.py
done
echo "Pipeline finished."
