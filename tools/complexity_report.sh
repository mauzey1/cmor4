#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  tools/complexity_report.sh [PATH ...]

Prints radon cyclomatic-complexity blocks of grade C or worse (complexity
above 10) and the maintainability index for each module. PATH defaults to the
cmor package. Run it before and after a change and compare the output.

Install radon with: pip install -e ".[dev]"

Options:
  -h, --help  Show this help
EOF
}

die() { echo "error: $*" >&2; exit 1; }

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
  -*)
    die "unknown argument: $1 (use --help)"
    ;;
esac

command -v radon >/dev/null 2>&1 || die "radon not found in PATH (pip install -e \".[dev]\")"

cd "${repo_root}"
if [[ $# -gt 0 ]]; then
  paths=("$@")
else
  paths=("cmor")
fi

echo "== Cyclomatic complexity, grade C or worse (radon cc -s -nC)"
radon cc -s -nC "${paths[@]}"
echo
echo "== Maintainability index (radon mi -s)"
radon mi -s "${paths[@]}"
