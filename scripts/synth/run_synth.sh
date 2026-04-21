#!/usr/bin/env bash
set -euo pipefail
docker exec vivado bash -c "source /home/user/Xilinx/2025.2/Vivado/settings64.sh && mkdir -p /home/user/red_eyes_is_all_you_need/build && vivado -mode batch -source /home/user/red_eyes_is_all_you_need/scripts/synth/synth.tcl"