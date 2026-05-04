#!/usr/bin/env bash
# Submit eval/submit_test_eval_matrix.sh with a custom SLURM --array (overrides the
# #SBATCH --array=0-61 line in that script). Use after partial runs / cancels.
#
# Usage (from repo root is typical; this script resolves paths from its location):
#   ./eval/submit_test_eval_matrix_rerun.sh
#       → default: tasks 7, 12, 18, and 29–61 only
#   ./eval/submit_test_eval_matrix_rerun.sh '0-3,10'
#   ./eval/submit_test_eval_matrix_rerun.sh '7,12,18,29-61' 'afterok:12345678'
#       → also wait until job 12345678 finishes successfully (e.g. last training job)
#
# Chain heatmaps after this eval array finishes (substitute the Job ID sbatch prints):
#   sbatch --dependency=afterok:<EVAL_JOBID> eval/submit_plot_heatmaps.sh
# If some array tasks may fail but you still want the CPU plot job to run:
#   sbatch --dependency=afterany:<EVAL_JOBID> eval/submit_plot_heatmaps.sh
#
# Note: afterok waits for all listed array tasks to exit 0; skipped tasks exit 0.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SH="${SCRIPT_DIR}/submit_test_eval_matrix.sh"

DEFAULT_ARRAY="7,12,18,29-61"
ARRAY_SPEC="${1:-$DEFAULT_ARRAY}"
DEP_SPEC="${2:-}"

if [[ ! -f "$EVAL_SH" ]]; then
  echo "ERROR: missing ${EVAL_SH}" >&2
  exit 2
fi

cmd=(sbatch --array="${ARRAY_SPEC}")
if [[ -n "${DEP_SPEC}" ]]; then
  cmd+=("--dependency=${DEP_SPEC}")
fi
cmd+=("${EVAL_SH}")

echo "==> ${cmd[*]}" >&2
out="$("${cmd[@]}" 2>&1)" || {
  echo "${out}" >&2
  exit 1
}
echo "${out}"
if [[ "${out}" =~ Submitted[[:space:]]+batch[[:space:]]+job[[:space:]]+([0-9]+) ]]; then
  jid="${BASH_REMATCH[1]}"
  echo "" >&2
  echo "Heatmaps chained after this eval (all tasks must succeed for afterok):" >&2
  echo "  sbatch --dependency=afterok:${jid} ${SCRIPT_DIR}/submit_plot_heatmaps.sh" >&2
  echo "Heatmaps after eval regardless of per-task exit code:" >&2
  echo "  sbatch --dependency=afterany:${jid} ${SCRIPT_DIR}/submit_plot_heatmaps.sh" >&2
fi
