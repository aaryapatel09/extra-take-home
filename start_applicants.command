#!/bin/zsh
set -eu
cd -- "${0:A:h}"
if [[ ! -x .venv/bin/python ]]; then
  echo "Create .venv and install requirements.txt first; see APPLICANTS.md." >&2
  exit 1
fi
exec .venv/bin/python applicant_cli.py "$@"
