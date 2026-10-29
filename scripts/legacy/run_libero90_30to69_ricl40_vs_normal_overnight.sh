#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PREPROCESSING_DIR="${PREPROCESSING_DIR:-ricl_libero_preprocessing/libero90_official30to69_leftout10_seeded40}"
export RETRIEVAL_EPISODES_PER_TASK="${RETRIEVAL_EPISODES_PER_TASK:-40}"
export RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
export RICL_VIDEO_OUT_DIR="${RICL_VIDEO_OUT_DIR:-data/libero/videos/libero90_30to69_ricl40_leftout10_${RUN_TAG}}"
export NORMAL_VIDEO_OUT_DIR="${NORMAL_VIDEO_OUT_DIR:-data/libero/videos/libero90_30to69_normal_leftout10_${RUN_TAG}}"
export REPORT_XLSX="${REPORT_XLSX:-data/libero/reports/libero90_30to69_ricl40_vs_normal_${RUN_TAG}.xlsx}"
export MANIFEST_JSON="${MANIFEST_JSON:-data/libero/reports/libero90_30to69_ricl40_vs_normal_${RUN_TAG}.manifest.json}"

exec "${SCRIPT_DIR}/run_libero90_30to69_ricl20_vs_normal_overnight.sh" "$@"
