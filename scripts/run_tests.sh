#!/usr/bin/env bash
# Run all testbench simulations in Docker and validate results
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

# Discover all testbenches from tb/ directory
tb_files=$(docker exec "$CONTAINER" bash -c "ls $PROJ/tb/tb_*.v")

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

for script in "$PROJ_DIR"/scripts/tests/validate_*.py; do
  base=$(basename "$script")
  val_total=$((val_total + 1))
  echo ">>> $base"

  if output=$(python3 "$script" 2>&1); then
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