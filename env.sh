#!/usr/bin/env bash

# Source this file before running project commands:
#   source env.sh
# It resolves paths relative to this file, so it works after moving the repo.

set -e

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Please source this file instead of executing it:"
  echo "  source $0"
  exit 1
fi

_env_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MAS_LSLM_ROOT="${_env_script_dir}"

case ":${PYTHONPATH:-}:" in
  *":${MAS_LSLM_ROOT}:"*) ;;
  *) export PYTHONPATH="${MAS_LSLM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac

cd "${MAS_LSLM_ROOT}"

echo "MAS_LSLM_ROOT=${MAS_LSLM_ROOT}"
