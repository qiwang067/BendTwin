#!/usr/bin/env bash
# Compute material hashes from hyperparameters; supports one or multiple semicolon-separated configurations.
#
# Usage (single configuration via command-line arguments):
#   bash compute_hash.sh --init_spring_Y 6e4 --spring_Y_min 5e4 --spring_Y_max 1e5 \
#                        --bend_stiffness 7 --bend_stiffness_min 0 --bend_stiffness_max 15
#
# Usage (multiple configurations via an environment variable):
#   STIFFNESS_CONFIGS="6e4,5e4,1e5,7,0,15;2e4,0,6e4,75,50,100" bash compute_hash.sh
#
# Usage (single configuration via an environment variable):
#   STIFFNESS_CONFIGS="6e4,5e4,1e5,7,0,15" bash compute_hash.sh
#
# The six values must use the following order:
#   init_spring_Y,spring_Y_min,spring_Y_max,bend_stiffness,bend_stiffness_min,bend_stiffness_max

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

compute_hash_single() {
    local init_spring_Y=$1
    local spring_Y_min=$2
    local spring_Y_max=$3
    local bend_stiffness=$4
    local bend_stiffness_min=$5
    local bend_stiffness_max=$6

    python3 "${script_dir}/material_utils.py" hash \
        --init_spring_Y "${init_spring_Y}" \
        --spring_Y_min "${spring_Y_min}" \
        --spring_Y_max "${spring_Y_max}" \
        --bend_stiffness "${bend_stiffness}" \
        --bend_stiffness_min "${bend_stiffness_min}" \
        --bend_stiffness_max "${bend_stiffness_max}"
}

# --- Command-line argument mode ---
if [ $# -gt 0 ]; then
    init_spring_Y=""
    spring_Y_min=""
    spring_Y_max=""
    bend_stiffness=""
    bend_stiffness_min=""
    bend_stiffness_max=""

    while [ $# -gt 0 ]; do
        case "$1" in
            --init_spring_Y)    init_spring_Y=$2;    shift 2 ;;
            --spring_Y_min)     spring_Y_min=$2;     shift 2 ;;
            --spring_Y_max)     spring_Y_max=$2;     shift 2 ;;
            --bend_stiffness)   bend_stiffness=$2;   shift 2 ;;
            --bend_stiffness_min) bend_stiffness_min=$2; shift 2 ;;
            --bend_stiffness_max) bend_stiffness_max=$2; shift 2 ;;
            -h|--help)
                sed -n '2,/^$/p' "$0"
                exit 0
                ;;
            *)
                echo "Unknown argument: $1" >&2
                exit 1
                ;;
        esac
    done

    for var in init_spring_Y spring_Y_min spring_Y_max bend_stiffness bend_stiffness_min bend_stiffness_max; do
        if [ -z "${!var}" ]; then
            echo "Missing required argument: --${var}" >&2
            exit 1
        fi
    done

    hash="$(compute_hash_single "${init_spring_Y}" "${spring_Y_min}" "${spring_Y_max}" \
                                "${bend_stiffness}" "${bend_stiffness_min}" "${bend_stiffness_max}")"
    printf 'hash=%s  (%s,%s,%s,%s,%s,%s)\n' \
        "${hash}" "${init_spring_Y}" "${spring_Y_min}" "${spring_Y_max}" \
        "${bend_stiffness}" "${bend_stiffness_min}" "${bend_stiffness_max}"
    exit 0
fi

# --- Environment-variable mode (supports multiple configurations) ---
stiffness_configs="${STIFFNESS_CONFIGS:-}"
if [ -z "${stiffness_configs}" ]; then
    echo "Usage: bash $0 --init_spring_Y <v> --spring_Y_min <v> --spring_Y_max <v> \\" >&2
    echo "             --bend_stiffness <v> --bend_stiffness_min <v> --bend_stiffness_max <v>" >&2
    echo "Or:    STIFFNESS_CONFIGS=\"6e4,5e4,1e5,7,0,15;2e4,0,6e4,75,50,100\" bash $0" >&2
    exit 1
fi

IFS=';' read -r -a entries <<< "${stiffness_configs}"
for entry in "${entries[@]}"; do
    entry="${entry//,/ }"
    read -r init_spring_Y spring_Y_min spring_Y_max bend_stiffness bend_stiffness_min bend_stiffness_max extra <<< "${entry}"
    if [ -z "${init_spring_Y}" ]; then
        continue
    fi
    if [ -n "${extra:-}" ]; then
        echo "Each configuration must contain exactly six values: ${entry}" >&2
        exit 1
    fi
    hash="$(compute_hash_single "${init_spring_Y}" "${spring_Y_min}" "${spring_Y_max}" \
                                "${bend_stiffness}" "${bend_stiffness_min}" "${bend_stiffness_max}")"
    printf 'hash=%s  (%s,%s,%s,%s,%s,%s)\n' \
        "${hash}" "${init_spring_Y}" "${spring_Y_min}" "${spring_Y_max}" \
        "${bend_stiffness}" "${bend_stiffness_min}" "${bend_stiffness_max}"
done
