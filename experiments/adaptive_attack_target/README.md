# Adaptive-attack target: Llama-2-7B-Chat (Narval)

Seen-family fine-tune: **λ=0.1, ε=-0.5, lr=2e-5, 5 epochs**, clean LM loss,
`train_plus_validation.csv`.

## Submit on Narval

Five consecutive **3-hour** jobs (epochs 1→5), chained with `afterok`:

```bash
cd $SCRATCH/dp-llm-experiments
git pull   # or rsync
mkdir -p output
bash experiments/adaptive_attack_target/submit_narval_5ep.sh
```

## Where the checkpoint lands

```text
$SCRATCH/adaptive_attack_target/llama2_7b_chat_lam0.1_eps-0.5_finetuned_llm_epoch5
$SCRATCH/adaptive_attack_target/LATEST_epoch5          # symlink to epoch5
$SCRATCH/adaptive_attack_target/README_PATHS.txt
```

## Merge before GCG / AutoDAN / PAIR

Attacks need a full HF dir (not LoRA-only). After training finishes:

```bash
source $SCRATCH/venv/nanogcg/bin/activate
python - <<'PY'
import os
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base = os.path.expandvars("$SCRATCH/llama2_7b_chat_hf")
adapter = os.path.expandvars(
    "$SCRATCH/adaptive_attack_target/llama2_7b_chat_lam0.1_eps-0.5_finetuned_llm_epoch5"
)
out = os.path.expandvars("$SCRATCH/merged_adaptive_l2_lam0.1_eps-0.5_ep5")

tok = AutoTokenizer.from_pretrained(base)
m = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16, device_map="auto")
m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
m.save_pretrained(out)
tok.save_pretrained(out)
print("merged ->", out)
PY
```

Then point GCG/AutoDAN/PAIR at `$SCRATCH/merged_adaptive_l2_lam0.1_eps-0.5_ep5`.
