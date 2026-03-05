#!/usr/bin/env bash
# Run testbench simulations in Docker and validate results
#
# Usage:
#   ./run_tests.sh                  # run all testbenches
#   ./run_tests.sh fp16             # run tb_fp16 only
#   ./run_tests.sh attention softmax  # run tb_attention and tb_softmax
#   ./run_tests.sh transformer_layer_stress  # run stress test
set -euo pipefail

CONTAINER=vivado
VIVADO_SH="source /home/user/Xilinx/2025.2/Vivado/settings64.sh"
PROJ=/home/user/red_eyes_is_all_you_need

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"
VENV="$SCRIPT_DIR/train/venv/bin/activate"

sim_fails=0
val_fails=0
sim_total=0
val_total=0

# Build list of testbenches to run
if [ $# -gt 0 ]; then
  tb_files=""
  for name in "$@"; do
    tb_files="$tb_files $PROJ/tb/tb_${name}.v"
  done
else
  tb_files=$(docker exec "$CONTAINER" bash -c "ls $PROJ/tb/tb_*.v")
fi

echo "Running simulations"
echo ""

for tb_path in $tb_files; do
  tb_file=$(basename "$tb_path" .v)
  sim_total=$((sim_total + 1))
  echo ">>> $tb_file"

  output=$(docker exec "$CONTAINER" bash -c "
    $VIVADO_SH
    rm -rf /tmp/xsim_$tb_file && mkdir -p /tmp/xsim_$tb_file && cd /tmp/xsim_$tb_file
    xvlog -i $PROJ/rtl $PROJ/rtl/*.v $PROJ/tb/${tb_file}.v 2>&1 && \
    xelab -debug off -timescale 1ns/1ps work.$tb_file -s snap 2>&1 && \
    xsim snap -runall 2>&1
  " 2>&1) || true

  if echo "$output" | grep -qiE "^(ERROR|FAIL|TIMEOUT)"; then
    echo "  FAIL"
    echo "$output" | grep -iE "^(ERROR|FAIL|TIMEOUT)"
    sim_fails=$((sim_fails + 1))
  else
    echo "  PASS"
  fi
done

echo ""
echo "Running validation scripts"
echo ""

source "$VENV"

# Map tb names to validation scripts
# Stress TBs use the base validator with the stress log path
declare -A VAL_MAP
VAL_MAP=(
  [tb_fp16]=validate_fp16.py
  [tb_attention]=validate_attention.py
  [tb_attention_stress]="validate_attention.py logs/tb_attention_stress.log"
  [tb_embedding]=validate_embedding.py
  [tb_gelu]=validate_gelu.py
  [tb_kv_cache]=validate_kv_cache.py
  [tb_layernorm]=validate_layernorm.py
  [tb_softmax]=validate_softmax.py
  [tb_transformer_layer]=validate_transformer_layer.py
  [tb_transformer_layer_stress]="validate_transformer_layer.py logs/tb_transformer_layer_stress.log"
  [tb_transformer_top]=validate_transformer_top.py
  [tb_transformer_top_stress]="validate_transformer_top.py logs/tb_transformer_top_stress.log"
  [tb_weight_store]=validate_weights.py
)

for tb_path in $tb_files; do
  tb_file=$(basename "$tb_path" .v)
  val_entry="${VAL_MAP[$tb_file]:-}"
  if [ -z "$val_entry" ]; then
    echo ">>> $tb_file: no validator, skipping"
    continue
  fi

  # Split into script and optional args
  script="${val_entry%% *}"
  args="${val_entry#"$script"}"
  script_path="$PROJ_DIR/scripts/tests/$script"

  if [ ! -f "$script_path" ]; then
    echo ">>> $script: not found, skipping"
    continue
  fi

  val_total=$((val_total + 1))
  echo ">>> $script${args}"

  if output=$(python3 "$script_path" $args 2>&1); then
    echo "  $(echo "$output" | tail -1)"
  else
    echo "  FAIL"
    echo "$output" | tail -3
    val_fails=$((val_fails + 1))
  fi
done

echo ""
total_fails=$((sim_fails + val_fails))
if [ $total_fails -eq 0 ]; then
  echo "=== All $sim_total sims + $val_total validations passed ==="
else
  echo "=== $total_fails failures ($sim_fails sim, $val_fails val) ==="
  exit 1
fi