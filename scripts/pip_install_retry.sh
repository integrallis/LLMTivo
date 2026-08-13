#!/usr/bin/env bash
# Install a package that may have JUST been published, retrying while the index catches up.
#
# Test PyPI does not make a new version resolvable the instant it is uploaded, and the delay is
# uneven: one runner installs it immediately while another 404s for a minute. Every place that
# installs a freshly published version needs the same patience, so it lives here once — three
# separate inline copies is how one of them ended up with no retry at all and failed a release.
#
#   scripts/pip_install_retry.sh "llmtivo[all]==0.1.2" [extra uv pip install args...]
set -euo pipefail

SPEC="$1"; shift
ATTEMPTS="${RETRY_ATTEMPTS:-5}"

for attempt in $(seq 1 "$ATTEMPTS"); do
  echo "Attempt $attempt of $ATTEMPTS: $SPEC"
  if uv pip install --system \
      --index-strategy unsafe-best-match \
      --extra-index-url https://test.pypi.org/simple/ \
      "$@" "$SPEC"; then
    echo "✅ installed $SPEC"
    exit 0
  fi
  if [ "$attempt" -lt "$ATTEMPTS" ]; then
    delay=$((attempt * 15))
    echo "⏳ not resolvable yet, waiting ${delay}s..."
    sleep "$delay"
  fi
done

echo "❌ $SPEC never became installable from Test PyPI after $ATTEMPTS attempts"
exit 1
