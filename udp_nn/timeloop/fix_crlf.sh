#!/usr/bin/env bash
# Fix Windows CRLF line endings on shell scripts (run once in WSL if you see:
#   env: bash\r: No such file or directory)
set -eu
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sed -i 's/\r$//' "$SCRIPT_DIR/run_mapper.sh" "$SCRIPT_DIR/wsl_run.sh" 2>/dev/null || true
chmod +x "$SCRIPT_DIR/run_mapper.sh"
echo "Fixed line endings in run_mapper.sh and wsl_run.sh"
