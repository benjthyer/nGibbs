#!/usr/bin/env bash
# hpc_fetch.sh -- copy one file from the HPC to a local destination. Run it LOCALLY.
#
# Usage:
#   ./hpc_fetch.sh /abs/path/on/hpc/file  [local_dest]  [--host user@login.hpc.edu]
#
#   local_dest  directory or file path (default: current directory)
#   --host      ssh target (or set HPC_HOST)
#
# Uses rsync --partial, so an interrupted transfer of a large (~20GB) file resumes
# where it left off: just re-run the same command. Network drops are retried automatically.

set -euo pipefail

HOST=${HPC_HOST:-bthyer@login.hpc.caltech.edu}
SRC=; DEST=.

while (($#)); do
  case $1 in
    --host)    HOST=${2:?missing value for $1}; shift 2 ;;
    -h|--help) sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)         if [[ -z $SRC ]]; then SRC=$1; else DEST=$1; fi; shift ;;
  esac
done
[[ -n $SRC ]] || { sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }

SSH="ssh -o ConnectTimeout=30 -o ServerAliveInterval=30 -o ServerAliveCountMax=6"

for i in $(seq 1 20); do
  if rsync --partial --append-verify --info=progress2 -e "$SSH" "$HOST:$SRC" "$DEST"; then
    echo "done: $DEST"; exit 0
  fi
  echo "rsync failed (attempt $i/20); retrying in 30s"; sleep 30
done
echo "giving up; re-run the same command to resume" >&2; exit 1
