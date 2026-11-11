#!/usr/bin/env bash
# WSL launcher — run from PowerShell:
#   wsl -e bash /home/hazem/run_timeloop_fixed.sh
#
# One-time setup (if ~/udp_timeloop symlink missing):
#   wsl -u root -e bash -c "ln -sfn '/mnt/d/Uni work/Semester 8/BACHELOR/Codes/udp_nn/timeloop' /home/hazem/udp_timeloop"

export PATH="$HOME/timeloop/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$HOME/timeloop/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$HOME/udp_timeloop"
exec ./run_mapper.sh "${1:-all}"
