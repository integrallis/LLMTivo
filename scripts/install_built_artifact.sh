#!/usr/bin/env bash
# Install the EXACT wheel this run built, with dependencies resolved from PyPI.
#
# The Test PyPI round trip is still exercised — the publish step uploads there and must succeed —
# but validation no longer waits on that index to become resolvable. Its simple endpoint is served
# by independent CDN caches that disagree: minutes after a successful upload, the JSON variant
# listed the new version while the HTML variant did not, so a poll could pass while the resolver
# on the same runner still saw nothing. That propagation tests the CDN, not this package.
#
# Installing the built artifact tests what actually matters and is deterministic: the wheel's
# metadata, its declared dependencies, its extras, and its entry points.
#
#   scripts/install_built_artifact.sh 0.1.4 '[all]' [extra uv pip install args...]
set -euo pipefail

VERSION="$1"; EXTRAS="${2:-}"; shift 2 || shift 1

WHEEL="dist/llmtivo-${VERSION}-py3-none-any.whl"
if [ ! -f "$WHEEL" ]; then
  echo "❌ $WHEEL not found — did the build job upload the test-dist artifact?"
  ls -la dist/ 2>/dev/null || echo "   (no dist/ directory)"
  exit 1
fi

echo "installing ${WHEEL}${EXTRAS}"
uv pip install --system "$@" "${WHEEL}${EXTRAS}"
