#!/usr/bin/env bash
set -u

# Read-only inventory for likely shared model locations. It intentionally avoids
# scanning all of /public and /work5, which contain petabytes of shared data.
roots=(
  /public/ai_data
  /public/DL_DATA
  /public/dtk
  /public/SothisAI
  /public/appmarket
  /public/software
)

report="${1:-model_inventory.txt}"
: > "$report"

{
  echo "Model inventory generated at: $(date -Is)"
  echo "Host: $(hostname)"
  echo "User: $USER"
  echo
  echo "=== Candidate files and directories ==="
} >> "$report"

for root in "${roots[@]}"; do
  if [[ -d "$root" ]]; then
    echo "--- $root ---" >> "$report"
    timeout 30s find "$root" -maxdepth 6 \
      \( -iname '*qwen*' -o -iname '*llama*' -o -iname '*chatglm*' \
         -o -iname '*baichuan*' -o -iname '*deepseek*' \
         -o -name 'model.safetensors.index.json' -o -name 'config.json' \) \
      -print 2>/dev/null | head -n 300 >> "$report"
  fi
done

{
  echo
  echo "=== Qwen-compatible config summaries ==="
} >> "$report"

while IFS= read -r config; do
  [[ -f "$config" ]] || continue
  if grep -qiE 'qwen|Qwen2ForCausalLM' "$config" 2>/dev/null; then
    model_dir=$(dirname "$config")
    echo "--- $model_dir ---" >> "$report"
    du -sh "$model_dir" 2>/dev/null >> "$report"
    grep -E '"model_type"|"architectures"|"hidden_size"|"num_hidden_layers"|"num_attention_heads"|"vocab_size"' \
      "$config" 2>/dev/null >> "$report"
    find "$model_dir" -maxdepth 1 -type f \
      \( -name '*.safetensors' -o -name '*.bin' -o -name 'tokenizer*' -o -name 'generation_config.json' \) \
      -printf '%f %s bytes\n' 2>/dev/null | sort >> "$report"
  fi
done < <(grep -E '/config\.json$' "$report" | sort -u)

{
  echo
  echo "=== Relevant modules ==="
  module avail 2>&1 | grep -Ei 'rocm|dtk|dcu|pytorch|torch|python|vllm|conda|apptainer' || true
  echo
  echo "=== Slurm resources ==="
  sinfo -o '%P|%a|%l|%D|%G' 2>&1 || true
} >> "$report"

echo "Wrote $report"
