#!/usr/bin/env bash
# Auto-download and exec ripvec-mcp for the current platform.
#
# Downloads the latest release binary on first use, caches in
# ${CLAUDE_PLUGIN_DATA}/bin/ (persists across plugin updates).
# Checks for new releases once per day.

set -euo pipefail

REPO="fnordpig/ripvec"

# Cache binary in CLAUDE_PLUGIN_DATA (persists across plugin version updates).
# Claude Code sets this to ~/.claude/plugins/data/<plugin>-<marketplace>/.
# Fallback chain: ~/.cache/ripvec → CLAUDE_PLUGIN_ROOT → script dir.
if [[ -n "${CLAUDE_PLUGIN_DATA:-}" ]]; then
	BIN_DIR="${CLAUDE_PLUGIN_DATA}/bin"
elif [[ -n "${HOME:-}" ]]; then
	BIN_DIR="${HOME}/.cache/ripvec/bin"
elif [[ -n "${CLAUDE_PLUGIN_ROOT:-}" ]]; then
	BIN_DIR="${CLAUDE_PLUGIN_ROOT}/bin"
else
	BIN_DIR="$(cd "$(dirname "$0")" && pwd)"
fi
mkdir -p "$BIN_DIR"

BINARY="${BIN_DIR}/ripvec-mcp"
VERSION_FILE="${BIN_DIR}/.version"
LAST_CHECK_FILE="${BIN_DIR}/.last-check"

# Detect platform
OS="$(uname -s)"
ARCH="$(uname -m)"

case "${OS}-${ARCH}" in
Darwin-arm64) TARGET="aarch64-apple-darwin" ;;
Darwin-x86_64) TARGET="aarch64-apple-darwin" ;;
Linux-x86_64) TARGET="x86_64-unknown-linux-gnu" ;;
Linux-aarch64) TARGET="aarch64-unknown-linux-gnu" ;;
*)
	echo "Unsupported platform: ${OS}-${ARCH}" >&2
	exit 1
	;;
esac

# Auto-detect CUDA
if [[ "$OS" == "Linux" ]]; then
	if [[ "${RIPVEC_CUDA:-auto}" == "auto" ]]; then
		command -v nvidia-smi &>/dev/null && TARGET="${TARGET}-cuda"
	elif [[ "${RIPVEC_CUDA:-}" == "1" ]]; then
		TARGET="${TARGET}-cuda"
	fi
fi

# --update: force immediate version check, bypassing cache
if [[ "${1:-}" == "--update" ]]; then
	rm -f "$LAST_CHECK_FILE"
	shift
fi

# --install-only: download binary without exec (used by hooks)
INSTALL_ONLY=false
if [[ "${1:-}" == "--install-only" ]]; then
	INSTALL_ONLY=true
	shift
fi

# Fetch latest version from GitHub API (~200ms, no cache)
get_latest_version() {
	local tag=""
	if command -v curl &>/dev/null; then
		tag=$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" 2>/dev/null |
			grep -o '"tag_name": *"[^"]*"' | head -1 | cut -d'"' -f4)
	elif command -v wget &>/dev/null; then
		tag=$(wget -qO- "https://api.github.com/repos/${REPO}/releases/latest" 2>/dev/null |
			grep -o '"tag_name": *"[^"]*"' | head -1 | cut -d'"' -f4)
	fi

	local version="${tag#v}"
	if [[ -n "$version" ]]; then
		echo "$version"
	elif [[ -f "$VERSION_FILE" ]]; then
		# Offline fallback: use whatever is installed
		cut -d: -f1 "$VERSION_FILE"
	else
		echo ""
	fi
}

RIPVEC_VERSION=$(get_latest_version)

if [[ -z "$RIPVEC_VERSION" ]]; then
	echo "Cannot determine ripvec version." >&2
	exit 1
fi

EXPECTED="${RIPVEC_VERSION}:${TARGET}"

# Fast path: binary exists, version+target match
if [[ -x "$BINARY" ]] && [[ -f "$VERSION_FILE" ]] && [[ "$(cat "$VERSION_FILE")" == "$EXPECTED" ]]; then
	$INSTALL_ONLY && exit 0
	exec "$BINARY" "$@"
fi

# Download
ARCHIVE="ripvec-v${RIPVEC_VERSION}-${TARGET}.tar.gz"
URL="https://github.com/${REPO}/releases/download/v${RIPVEC_VERSION}/${ARCHIVE}"

echo "ripvec-mcp v${RIPVEC_VERSION} — downloading for ${TARGET}..." >&2

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

if command -v curl &>/dev/null; then
	curl -fsSL "$URL" -o "${TMPDIR}/${ARCHIVE}"
elif command -v wget &>/dev/null; then
	wget -q "$URL" -O "${TMPDIR}/${ARCHIVE}"
else
	echo "No curl/wget. Install: cargo binstall ripvec-mcp" >&2
	exit 1
fi

tar xzf "${TMPDIR}/${ARCHIVE}" -C "$TMPDIR"
EXTRACT_DIR="${TMPDIR}/ripvec-v${RIPVEC_VERSION}-${TARGET}"

cp "${EXTRACT_DIR}/ripvec-mcp" "$BINARY"
chmod +x "$BINARY"
[[ -f "${EXTRACT_DIR}/ripvec" ]] && cp "${EXTRACT_DIR}/ripvec" "${BIN_DIR}/ripvec" && chmod +x "${BIN_DIR}/ripvec"

# macOS: the release binary ships ad-hoc linker-signed, which Gatekeeper
# silently SIGKILLs when launched from a non-TTY parent (Claude Code's
# MCP stdio child). Strip the quarantine xattr and re-ad-hoc-sign under
# the current user's trust scope so the binary actually runs.
if [[ "$OS" == "Darwin" ]] && command -v codesign &>/dev/null; then
	xattr -dr com.apple.quarantine "$BINARY" 2>/dev/null || true
	codesign --force --sign - "$BINARY" 2>/dev/null || true
	if [[ -x "${BIN_DIR}/ripvec" ]]; then
		xattr -dr com.apple.quarantine "${BIN_DIR}/ripvec" 2>/dev/null || true
		codesign --force --sign - "${BIN_DIR}/ripvec" 2>/dev/null || true
	fi
fi

echo "${RIPVEC_VERSION}:${TARGET}" >"$VERSION_FILE"
echo "ripvec-mcp v${RIPVEC_VERSION} installed." >&2

# Preflight: surface silent-launch failures (missing shared libs, glibc
# mismatch, CUDA runtime absent) before Claude Code sees "connection closed"
# with zero stderr and reports "Failed to reconnect".
if [[ "$OS" == "Linux" ]] && command -v timeout &>/dev/null; then
	PREFLIGHT=$(timeout 2 "$BINARY" </dev/null 2>&1 >/dev/null || true)
	# Filter out the benign MCP-serve error that fires when stdin is /dev/null;
	# anything else (linker errors, glibc mismatches, CUDA load failures) is
	# a real launch problem.
	REAL_ERR=$(echo "$PREFLIGHT" | grep -v -iE 'MCP serve|connection closed' || true)
	if [[ -n "$REAL_ERR" ]]; then
		echo "ripvec-mcp preflight failed:" >&2
		echo "$REAL_ERR" >&2
		if echo "$REAL_ERR" | grep -qiE 'libcud|libnvrtc|libcublas'; then
			echo "" >&2
			echo "Hint: CUDA runtime not found. Set RIPVEC_CUDA=0 for the CPU-only build." >&2
		fi
		if echo "$REAL_ERR" | grep -qiE 'libopenblas|libblas'; then
			echo "" >&2
			echo "Hint: install OpenBLAS — sudo apt-get install libopenblas0 (or equivalent)" >&2
		fi
		if echo "$REAL_ERR" | grep -qiE 'GLIBC_[0-9.]+.*not found'; then
			echo "" >&2
			echo "Hint: release binaries are built against glibc from ubuntu-latest." >&2
			echo "Older distros may need to build from source: cargo install ripvec-mcp" >&2
		fi
		exit 1
	fi
fi

$INSTALL_ONLY && exit 0
exec "$BINARY" "$@"
