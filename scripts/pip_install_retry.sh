#!/usr/bin/env bash
# Install a package that may have JUST been published to Test PyPI.
#
# Uploading is not publishing. The upload returns 200 OK and the file is accepted, but the index
# that resolvers read is eventually consistent and can lag by minutes — unevenly, so one runner
# installs immediately while another 404s. Retrying the INSTALL conflates two questions ("is it
# indexed yet?" and "does it install?") and reports an unresolvable-dependency error for what is
# really a wait.
#
# So: wait for the version to appear in the simple index, with a real deadline, and only then
# install. A timeout here means the index genuinely never caught up, which is worth failing on.
#
#   scripts/pip_install_retry.sh "llmtivo[all]==0.1.3" [extra uv pip install args...]
set -euo pipefail

SPEC="$1"; shift
DEADLINE_S="${INDEX_DEADLINE_S:-600}"
INDEX="${TEST_PYPI_SIMPLE:-https://test.pypi.org/simple/llmtivo/}"

# "llmtivo[all]==0.1.3" -> 0.1.3
VERSION="${SPEC##*==}"

echo "waiting for llmtivo $VERSION in $INDEX (deadline ${DEADLINE_S}s)"
waited=0
until curl -fsSL "$INDEX" 2>/dev/null | grep -q "llmtivo-${VERSION}"; do
  if [ "$waited" -ge "$DEADLINE_S" ]; then
    echo "❌ llmtivo $VERSION never appeared in the index after ${DEADLINE_S}s"
    echo "   (the upload may have succeeded — check the publish step — but resolvers cannot see it)"
    exit 1
  fi
  sleep 10
  waited=$((waited + 10))
  if [ $((waited % 60)) -eq 0 ]; then echo "  still waiting... ${waited}s"; fi
done
echo "✅ indexed after ${waited}s"

# --refresh-package: uv caches index metadata, and setup-uv RESTORES that cache between runs, so a
# version published seconds ago stays invisible to the resolver even though the index itself serves
# it — the poll above proves the index is fine while `uv pip install` still reports it unresolvable.
# Refreshing just this package keeps every other dependency served from cache.
uv pip install --system \
  --refresh-package llmtivo \
  --index-strategy unsafe-best-match \
  --extra-index-url https://test.pypi.org/simple/ \
  "$@" "$SPEC"
