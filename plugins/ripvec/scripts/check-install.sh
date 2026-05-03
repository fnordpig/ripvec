#!/bin/bash
# Compatibility shim for older local plugin installs.

set -eo pipefail

root="${CLAUDE_PLUGIN_ROOT}"
case "$root" in ''|*'$'*) root="$(cd "$(dirname "$0")/.." && pwd)" ;; esac
exec bash "$root/hooks/scripts/check-install.sh"
