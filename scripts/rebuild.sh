#!/usr/bin/env bash
# Rebuild and reinstall the wheel.
#
# The Python sources are packaged *inside* the wheel, so editing a .py file in
# python/mixedlm_rs/ has no effect on the installed package until this runs.
# Forgetting that reads as "my fix did not work".
set -euo pipefail
cd "$(dirname "$0")/.."
python -m maturin build --release --out dist "$@" >/dev/null
wheel="$(ls -t dist/*.whl | head -1)"
python -m pip install --force-reinstall --no-deps -q "$wheel"
echo "installed $wheel"
