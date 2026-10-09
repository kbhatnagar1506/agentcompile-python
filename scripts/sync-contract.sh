#!/usr/bin/env bash
# Vendor the SDK <-> server contract (golden requests/responses and their schemas) from the
# AgentCompile server repo into tests/contract/.
#
#   scripts/sync-contract.sh                       # from ../agent-compiler (a local checkout)
#   AGENT_COMPILER_DIR=/path/to/agent-compiler scripts/sync-contract.sh
#   AGENT_COMPILER_REF=main scripts/sync-contract.sh   # fetch by ref with gh (needs repo access)
#   scripts/sync-contract.sh --check               # fail if the vendored copy differs
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
dest="$here/tests/contract"
check=0
[[ "${1:-}" == "--check" ]] && check=1

src="${AGENT_COMPILER_DIR:-$here/../agent-compiler}/tests/contract"
tmp=""
if [[ -n "${AGENT_COMPILER_REF:-}" ]]; then
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  mkdir -p "$tmp/golden"
  repo="kbhatnagar1506/agent-compiler"
  get() { gh api -H "Accept: application/vnd.github.raw" "repos/$repo/contents/tests/contract/$1?ref=$AGENT_COMPILER_REF"; }
  get schemas.json > "$tmp/schemas.json"
  for name in $(gh api "repos/$repo/contents/tests/contract/golden?ref=$AGENT_COMPILER_REF" --jq '.[].name'); do
    get "golden/$name" > "$tmp/golden/$name"
  done
  src="$tmp"
fi
[[ -f "$src/schemas.json" ]] || { echo "no contract at $src (set AGENT_COMPILER_DIR or AGENT_COMPILER_REF)" >&2; exit 2; }

if [[ $check == 1 ]]; then
  diff -r "$src/golden" "$dest/golden" && diff "$src/schemas.json" "$dest/schemas.json" \
    || { echo "tests/contract is out of date with the server: run scripts/sync-contract.sh" >&2; exit 1; }
  echo "tests/contract matches the server"
  exit 0
fi
rm -rf "$dest/golden"
mkdir -p "$dest"
cp -r "$src/golden" "$dest/golden"
cp "$src/schemas.json" "$dest/schemas.json"
echo "synced $(ls "$dest/golden" | wc -l | tr -d ' ') golden files from $src"
