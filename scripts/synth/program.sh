#!/usr/bin/env bash
set -euo pipefail
PROJ_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
openFPGALoader --board alinx_ax7203 --cable digilent_hs2 "$PROJ_DIR/build/top.bit"