#!/usr/bin/env bash
# hpc_bundle_all.sh -- one command to: compress a huge directory on an HPC (SLURM),
# download it, verify it per-file against checksums of the ORIGINAL files, then
# delete the original on the HPC and clean up local scratch. Run it LOCALLY.
#
# Usage:
#   ./hpc_bundle_all.sh --host user@login.hpc.edu \
#       --hpc-dir   /abs/path/on/hpc/bigdir \
#       --dest      /mnt/hdd/bigdir_bundles \
#       --workspace /mnt/ssd/scratch \
#       [--hpc-scratch /abs/scratch/parent]   # default: /resnick/scratch/bthyer
#       [--yes]                                # skip the deletion confirmation prompt
#
#   --hpc-dir      directory to compress (and, at the end, delete). Absolute path.
#   --hpc-scratch  parent dir on the HPC for the compressed bundles; they go in
#                  <hpc-scratch>/bundle_<name>/ (removed after successful deletion step)
#   --dest         local destination; receives the .tar.zst bundles + checksum manifests
#   --workspace    local scratch for verification (a few archives' worth of space; temp files only)
#   --host         ssh target (or set HPC_HOST)
#
# Steps:
#   1. sbatch a job that compresses hpc_dir into hpc_scratch (+ sha256 of every ORIGINAL file)
#   2. wait for the job
#   3. rsync the bundles to dest; check archive checksums
#   4. extract archive-by-archive into workspace (dest untouched; each temp copy is deleted
#      right after hashing); compare every file's sha256, count and symlinks to the original
#   5. only if 4 passed: confirm, then sbatch a job that deletes hpc_dir (with safety checks)
#   6. delete the extracted copy in workspace
#
# Resumable: if interrupted, re-run the identical command. Env tunables:
#   SBATCH_OPTS         default "--cpus-per-task=8 --mem=16G --time=48:00:00"
#                       (add --partition/--account/--qos as your cluster needs)
#   DELETE_SBATCH_OPTS  default "--cpus-per-task=1 --mem=4G --time=24:00:00"
#   REMOTE_PRELUDE      shell commands run first inside the jobs, e.g. "module load zstd"
#   ZSTD_LEVEL (10)  CHUNK_FILES (200000 files/archive)  LOCAL_JOBS (4)  POLL_SECS (60)

set -euo pipefail

SBATCH_OPTS=${SBATCH_OPTS:---cpus-per-task=8 --mem=16G --time=48:00:00}
DELETE_SBATCH_OPTS=${DELETE_SBATCH_OPTS:---cpus-per-task=1 --mem=4G --time=24:00:00}
ZSTD_LEVEL=${ZSTD_LEVEL:-10}
CHUNK_FILES=${CHUNK_FILES:-200000}
LOCAL_JOBS=${LOCAL_JOBS:-4}
POLL_SECS=${POLL_SECS:-60}
REMOTE_PRELUDE=${REMOTE_PRELUDE:-}

HOST=${HPC_HOST:-bthyer@login.hpc.caltech.edu}; HPC_DIR=; HPC_SCRATCH=/resnick/scratch/bthyer; DEST=; WORKSPACE=; ASSUME_YES=0

log() { echo "[$(date '+%F %T')] $*"; }
err() { echo "[$(date '+%F %T')] ERROR: $*" >&2; }
die() { err "$*"; exit 1; }
usage() { sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
q() { printf '%q' "$1"; }

while (($#)); do
  case $1 in
    --host)         HOST=${2:?missing value for $1}; shift 2 ;;
    --hpc-dir)      HPC_DIR=${2:?missing value for $1}; shift 2 ;;
    --hpc-scratch)  HPC_SCRATCH=${2:?missing value for $1}; shift 2 ;;
    --dest)         DEST=${2:?missing value for $1}; shift 2 ;;
    --workspace)    WORKSPACE=${2:?missing value for $1}; shift 2 ;;
    --yes)          ASSUME_YES=1; shift ;;
    -h|--help)      usage ;;
    *)              die "unknown argument: $1 (see --help)" ;;
  esac
done
[[ -n $HOST && -n $HPC_DIR && -n $DEST && -n $WORKSPACE ]] || usage
[[ $HPC_DIR == /* ]] || die "--hpc-dir must be an absolute path (a leading ~ would expand to your LOCAL home)"
[[ -z $HPC_SCRATCH || $HPC_SCRATCH == /* ]] || die "--hpc-scratch must be an absolute path"
for t in rsync zstd tar split xargs df numfmt ssh; do command -v "$t" >/dev/null || die "local tool missing: $t"; done

HPC_DIR=${HPC_DIR%/}
DEST=$(realpath -m "$DEST"); WORKSPACE=$(realpath -m "$WORKSPACE")
[[ $DEST != "$WORKSPACE" ]] || die "--dest and --workspace must differ"
NAME=$(basename "$HPC_DIR")

mkdir -p "$HOME/.ssh"
SSH_OPTS=(-o ConnectTimeout=30 -o ServerAliveInterval=30 -o ServerAliveCountMax=6
          -o ControlMaster=auto -o "ControlPath=$HOME/.ssh/cm-bundle-%C" -o ControlPersist=6h)
rsh()    { ssh -n "${SSH_OPTS[@]}" "$HOST" "$@"; }            # no stdin
upload() { ssh "${SSH_OPTS[@]}" "$HOST" "cat > $(q "$1") && chmod +x $(q "$1")"; }  # stdin -> remote file

trap 'echo; log "interrupted -- safe to re-run the same command to resume"; exit 130' INT TERM

need_space() { # path bytes label
  local avail; avail=$(df --output=avail -B1 "$1" | tail -n1 | tr -d ' ')
  (( avail >= $2 )) || die "not enough space on $3 ($1): need $(numfmt --to=iec "$2"), have $(numfmt --to=iec "$avail")"
}
sum_sizes() { find "$1" -maxdepth 1 -name '*.tar.zst' -printf '%s\n' | awk '{s+=$1} END{printf "%d\n", s}'; }

# ------------------------------------------------------------------ remote job scripts
emit_pack_job() {
  printf '#!/usr/bin/env bash\n%s\n' "$REMOTE_PRELUDE"
  cat <<'PACKJOB'
set -euo pipefail
SRC=$1; STAGE=$2; LEVEL=$3; CHUNK=$4
THREADS=${SLURM_CPUS_PER_TASK:-8}
trap 'rc=$?; [[ $rc -eq 0 ]] || echo "FAIL rc=$rc" > "$STAGE/PACK_STATUS"' EXIT
mkdir -p "$STAGE/.lists" "$STAGE/.hash"
rm -f "$STAGE/PACK_STATUS" "$STAGE/PACK_COMPLETE"
[[ -f $STAGE/PACK_START ]] || touch "$STAGE/PACK_START"
command -v zstd >/dev/null || { echo "zstd not found on compute node (use REMOTE_PRELUDE='module load zstd')" >&2; exit 90; }
cd "$SRC"

if [[ ! -f $STAGE/.lists/guard_done ]]; then
  bad=$(find . \( -name $'*\n*' -o -name $'*\t*' \) -print -quit | head -c 300 || true)
  [[ -z $bad ]] || { echo "unsupported filename (contains newline/tab): $bad" >&2; exit 91; }
  touch "$STAGE/.lists/guard_done"
fi

groups=()
while IFS= read -r -d '' d; do groups+=("${d#./}"); done < <(find . -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
groups+=("__toplevel_files__")

for g in "${groups[@]}"; do
  safe=${g//\//_}; list="$STAGE/.lists/$safe.all"
  if [[ ! -f $STAGE/.lists/$safe.split_done ]]; then
    echo "listing: $g"
    rm -f "$STAGE/.lists/$safe".[0-9][0-9][0-9][0-9]
    if [[ $g == __toplevel_files__ ]]; then
      find . -mindepth 1 -maxdepth 1 ! -type d -print0 | sort -z > "$list"
    else
      find "./$g" ! -type d -print0 | sort -z > "$list"
    fi
    [[ ! -s $list ]] || split -d -a 4 -t '\0' -l "$CHUNK" "$list" "$STAGE/.lists/$safe."
    touch "$STAGE/.lists/$safe.split_done"
  fi
  for chunk in "$STAGE/.lists/$safe".[0-9][0-9][0-9][0-9]; do
    [[ -e $chunk ]] || continue
    name="$safe.${chunk##*.}.tar.zst"; out="$STAGE/$name"
    [[ -f $out.ok ]] && { echo "skip (done): $name"; continue; }
    expect=$(tr -cd '\0' < "$chunk" | wc -c)
    echo "packing $name ($expect entries)"
    rm -f "$out.partial"
    tar --null --no-recursion -I "zstd -T$THREADS -$LEVEL" -cf "$out.partial" -T "$chunk"
    actual=$(tar -I zstd -tf "$out.partial" | wc -l)
    [[ $actual -eq $expect ]] || { echo "count mismatch in $name: $expect vs $actual" >&2; exit 92; }
    mv "$out.partial" "$out"
    (cd "$STAGE" && sha256sum "$name" > "$name.sha256")
    printf '%s\t%s\n' "$name" "$expect" > "$out.count"
    touch "$out.ok"
  done
done

# Per-file checksums of the ORIGINAL regular files, and symlink targets.
if [[ ! -f $STAGE/.lists/hash_done ]]; then
  echo "hashing original files"
  export HASHDIR="$STAGE/.hash"
  find "$HASHDIR" -type f -delete
  find . -type f -print0 | xargs -0 -r -n 1000 -P "$THREADS" bash -c 'sha256sum -- "$@" > "$(mktemp -p "$HASHDIR" h.XXXXXX)"' _
  find "$HASHDIR" -type f -name 'h.*' -exec cat {} + > "$STAGE/MANIFEST.sha256.tmp"
  mv "$STAGE/MANIFEST.sha256.tmp" "$STAGE/MANIFEST.sha256"
  find . -type l -printf '%p\t%l\n' > "$STAGE/LINKS.tsv"
  touch "$STAGE/.lists/hash_done"
fi

cd "$STAGE"
cat ./*.tar.zst.sha256 > SHA256SUMS 2>/dev/null || : > SHA256SUMS
cat ./*.tar.zst.count  > COUNTS.tsv 2>/dev/null || : > COUNTS.tsv
awk -F'\t' '{n+=$2} END{printf "%d\n", n}' COUNTS.tsv > TOTAL_FILES
(cd "$SRC" && find . -type f -printf '%s\n') | awk '{s+=$1} END{printf "%d\n", s}' > TOTAL_BYTES
find . -maxdepth 1 -name '*.tar.zst' -printf '%s\n' | awk '{s+=$1} END{printf "%d\n", s}' > COMPRESSED_BYTES
date -u +%FT%TZ > PACK_COMPLETE
echo OK > PACK_STATUS
echo "pack complete: $(cat TOTAL_FILES) entries"
PACKJOB
}

emit_delete_job() {
  printf '#!/usr/bin/env bash\n%s\n' "$REMOTE_PRELUDE"
  cat <<'DELJOB'
set -uo pipefail
DIR=$1; STAGE=$2; SF=$3
fail() { echo "FAIL $*" > "$SF"; echo "ABORT: $*" >&2; exit 1; }
[[ $(basename "$STAGE") == bundle_* ]] || fail "unexpected stage name"
[[ -f $STAGE/PACK_COMPLETE ]] || fail "pack not complete"
if [[ ! -e $DIR ]]; then rm -rf "$STAGE"; echo OK > "$SF"; rm -f "$0"; exit 0; fi
REAL=$(realpath -e "$DIR") || fail "cannot resolve $DIR"
[[ $(tr -cd '/' <<<"$REAL" | wc -c) -ge 3 ]] || fail "refusing: path too shallow ($REAL)"
[[ $REAL != "$HOME" && $HOME/ != "$REAL"/* ]] || fail "refusing: path is or contains your home directory"
[[ $REAL/ != "$STAGE"/* && $STAGE/ != "$REAL"/* ]] || fail "refusing: stage and target overlap"
now=$(find "$REAL" ! -type d | wc -l)
[[ $now -eq $(cat "$STAGE/TOTAL_FILES") ]] || fail "entry count changed since packing ($now vs $(cat "$STAGE/TOTAL_FILES"))"
newer=$(find "$REAL" -newer "$STAGE/PACK_START" -print -quit)
[[ -z $newer ]] || fail "modified since packing started: $newer"
echo "deleting $REAL"
if ! find "$REAL" -mindepth 1 -delete; then
  chmod -R u+rwX "$REAL" 2>/dev/null
  find "$REAL" -mindepth 1 -delete || fail "delete failed"
fi
rmdir "$REAL" 2>/dev/null || echo "note: could not rmdir $REAL (left empty)"
rm -rf "$STAGE"
echo OK > "$SF"
rm -f "$0"
DELJOB
}

# ------------------------------------------------------------------ job helpers
# wait_job JOBID STATUSFILE LABEL  -> 0 if the job wrote OK, 1 otherwise
wait_job() {
  local jid=$1 sf=$2 label=$3 fails=0 gone=0 out state
  while true; do
    if ! out=$(rsh "cat $(q "$sf") 2>/dev/null; true"); then
      fails=$((fails + 1))
      (( fails <= 60 )) || { err "lost ssh connection to $HOST"; return 1; }
      log "ssh hiccup ($fails); retrying"; sleep "$POLL_SECS"; continue
    fi
    fails=0; out=$(tail -n1 <<<"$out")
    case $out in
      OK)    return 0 ;;
      FAIL*) err "$label job reported: $out (logs: *.out next to $STAGE on the HPC)"; return 1 ;;
    esac
    if state=$(rsh "squeue -h -j $jid -o %T 2>/dev/null; true"); then
      state=$(tail -n1 <<<"$state")
      if [[ -n $state ]]; then gone=0; log "$label job $jid: $state"
      else
        gone=$((gone + 1))
        (( gone < 3 )) || { err "$label job $jid left the queue without reporting success (cancelled / timed out?)"; return 1; }
      fi
    fi
    sleep "$POLL_SECS"
  done
}

submit() { # script_remote_path  status_file  sbatch_opts  jobname  outprefix  args...
  local script=$1 sf=$2 opts=$3 jname=$4 outp=$5; shift 5
  local args=""; local a; for a in "$@"; do args+=" $(q "$a")"; done
  local jid
  jid=$(rsh "rm -f $(q "$sf"); sbatch --parsable $opts --job-name=$jname --output=$(q "$outp")_%j.out $(q "$script")$args" | tail -n1) || return 1
  jid=${jid%%;*}
  [[ $jid =~ ^[0-9]+$ ]] || { err "sbatch did not return a job id: '$jid'"; return 1; }
  echo "$jid"
}

# ------------------------------------------------------------------ steps
preflight() {
  rsh "command -v sbatch >/dev/null" || die "cannot reach $HOST, or sbatch not found there"
  if [[ -z $HPC_SCRATCH ]]; then
    HPC_SCRATCH=$(rsh 'echo "${SCRATCH:-/scratch/$USER}"' | tail -n1)
    [[ $HPC_SCRATCH == /* ]] || die "could not determine HPC scratch; pass --hpc-scratch"
  fi
  STAGE="${HPC_SCRATCH%/}/bundle_$NAME"
  log "HPC dir:      $HOST:$HPC_DIR"
  log "HPC stage:    $STAGE"
  log "local dest:   $DEST"
  log "local work:   $WORKSPACE"
  rsh "command -v zstd >/dev/null" || log "note: zstd not on the login node PATH; if the job fails on that, set REMOTE_PRELUDE='module load zstd'"
  mkdir -p "$DEST" "$WORKSPACE"
}

PACK_JID=
step1_submit() {
  local st jid
  st=$(rsh "cat $(q "$STAGE/PACK_STATUS") 2>/dev/null; true" | tail -n1)
  if [[ $st == OK ]]; then log "step 1: bundles already packed on the HPC"; return; fi
  rsh "test -d $(q "$HPC_DIR")" || die "HPC directory not found: $HPC_DIR (and no finished bundles exist for it)"
  jid=$(rsh "cat $(q "$STAGE/JOB_ID") 2>/dev/null; true" | tail -n1)
  if [[ -n $jid ]] && [[ -n $(rsh "squeue -h -j $jid -o %T 2>/dev/null; true" | tail -n1) ]]; then
    log "step 1: attaching to existing pack job $jid"; PACK_JID=$jid; return
  fi
  rsh "mkdir -p $(q "$STAGE")"
  emit_pack_job | upload "$STAGE/pack_job.sh"
  PACK_JID=$(submit "$STAGE/pack_job.sh" "$STAGE/PACK_STATUS" "$SBATCH_OPTS" bundle_pack "$STAGE/pack" \
              "$HPC_DIR" "$STAGE" "$ZSTD_LEVEL" "$CHUNK_FILES") || die "could not submit pack job"
  rsh "echo $PACK_JID > $(q "$STAGE/JOB_ID")"
  log "step 1: submitted pack job $PACK_JID"
}

step2_wait() {
  if [[ -z $PACK_JID ]]; then return; fi
  log "step 2: waiting for pack job $PACK_JID (polling every ${POLL_SECS}s; safe to Ctrl-C and re-run)"
  wait_job "$PACK_JID" "$STAGE/PACK_STATUS" pack || die "pack failed"
  log "step 2: pack job finished OK"
}

step3_fetch() {
  local csz have need i ok=0
  csz=$(rsh "cat $(q "$STAGE/COMPRESSED_BYTES")" | tail -n1)
  have=$(sum_sizes "$DEST")
  need=$(( csz > have ? csz - have + 104857600 : 104857600 ))
  need_space "$DEST" "$need" "destination"
  log "step 3: downloading $(numfmt --to=iec "$csz") of bundles to $DEST"
  for ((i = 1; i <= 20; i++)); do
    if rsync -a --partial --info=progress2 -e "ssh ${SSH_OPTS[*]}" \
         --include='*.tar.zst' --include=SHA256SUMS --include=COUNTS.tsv --include=MANIFEST.sha256 \
         --include=LINKS.tsv --include=TOTAL_FILES --include=TOTAL_BYTES --include=COMPRESSED_BYTES \
         --include=PACK_COMPLETE --exclude='*' "$HOST:$STAGE/" "$DEST/"; then ok=1; break; fi
    log "rsync failed (attempt $i/20); retrying in 30s"; sleep 30
  done
  (( ok )) || die "download failed repeatedly; re-run to resume"
  [[ -f $DEST/PACK_COMPLETE ]] || die "PACK_COMPLETE missing in download"
  local sig
  sig=$(cd "$DEST" && { find . -maxdepth 1 -name '*.tar.zst' -printf '%f %s %T@\n' | sort; sha256sum SHA256SUMS; } | sha256sum | cut -d' ' -f1)
  if [[ -f $DEST/.archives_verified && $(cat "$DEST/.archives_verified") == "$sig" ]]; then
    log "step 3: archive checksums already verified for these exact files; skipping"
  else
    log "step 3: checking archive checksums (slow on a spinning disk; only done once)"
    (cd "$DEST" && sha256sum -c --quiet SHA256SUMS) || die "archive checksum mismatch in $DEST (delete the bad file(s) and re-run)"
    echo "$sig" > "$DEST/.archives_verified"
  fi
}

VWORK=
verify_one() { # archive; run in parallel by xargs. Needs exported VWORK.
  local a=$1 b tmp ok=0
  b=$(basename "$a")
  tmp=$(mktemp -d -p "$VWORK" ex.XXXXXX) || return 1
  if tar -I zstd -xf "$a" -C "$tmp" \
     && (cd "$tmp" && find . -type f -print0 | xargs -0 -r sha256sum --) > "$VWORK/out/$b.sha" \
     && (cd "$tmp" && find . ! -type d | wc -l) > "$VWORK/out/$b.cnt" \
     && (cd "$tmp" && find . -type l -printf '%p\t%l\n') > "$VWORK/out/$b.lnk"; then ok=1; fi
  chmod -R u+rwX "$tmp" 2>/dev/null; rm -rf "$tmp"
  (( ok )) || { echo "FAILED: $b" >&2; return 1; }
  echo "  checked $b" >&2
}
export -f verify_one

step4_verify() {
  VWORK="$WORKSPACE/verify_$NAME"; export VWORK
  local total_bytes n avg expect actual
  total_bytes=$(cat "$DEST/TOTAL_BYTES")
  n=$(find "$DEST" -maxdepth 1 -name '*.tar.zst' | wc -l)
  avg=$(( total_bytes / (n > 0 ? n : 1) ))
  need_space "$WORKSPACE" $(( LOCAL_JOBS * avg * 3 + 2147483648 )) "workspace"
  [[ -e $VWORK ]] && { chmod -R u+rwX "$VWORK" 2>/dev/null || true; rm -rf "$VWORK"; }
  mkdir -p "$VWORK/out"
  log "step 4: verifying $n archives, $LOCAL_JOBS at a time (temp space only; $DEST is not touched)"
  find "$DEST" -maxdepth 1 -name '*.tar.zst' -print0 | xargs -0 -P "$LOCAL_JOBS" -I{} bash -c 'verify_one "$1"' _ {} \
    || die "extract/hash failed for at least one archive (see FAILED lines above)"

  log "step 4: comparing against the ORIGINAL manifest"
  expect=$(cat "$DEST/TOTAL_FILES")
  actual=$(cat "$VWORK"/out/*.cnt | awk '{s+=$1} END{printf "%d\n", s}')
  [[ $actual -eq $expect ]] || die "entry count mismatch: extracted $actual, expected $expect"
  cat "$VWORK"/out/*.sha | LC_ALL=C sort -T "$VWORK" > "$VWORK/local.sha"
  LC_ALL=C sort -T "$VWORK" "$DEST/MANIFEST.sha256" > "$VWORK/orig.sha"
  cmp -s "$VWORK/local.sha" "$VWORK/orig.sha" \
    || { diff "$VWORK/orig.sha" "$VWORK/local.sha" | head -20; die "FILE CHECKSUM MISMATCH -- do NOT delete anything"; }
  cat "$VWORK"/out/*.lnk | LC_ALL=C sort > "$VWORK/local.lnk"
  LC_ALL=C sort "$DEST/LINKS.tsv" > "$VWORK/orig.lnk"
  cmp -s "$VWORK/local.lnk" "$VWORK/orig.lnk" || die "symlink mismatch -- do NOT delete anything"

  printf 'verified %s: %s entries, %s bytes\n' "$(date -u +%FT%TZ)" "$expect" "$total_bytes" > "$DEST/VERIFIED"
  log "step 4: VERIFIED ($expect entries, $(numfmt --to=iec "$total_bytes"))"
}

step5_delete() {
  local exists ans jid
  exists=$(rsh "test -e $(q "$HPC_DIR") && echo yes || echo no" | tail -n1) || { err "ssh failed"; return 1; }
  if [[ $exists == no ]]; then log "step 5: HPC directory already gone; nothing to delete"; return 0; fi

  if (( ! ASSUME_YES )); then
    if [[ ! -t 0 ]]; then err "step 5 skipped: not interactive. Re-run with --yes to delete $HPC_DIR"; return 1; fi
    echo
    echo "Everything verified. The ONLY other copy of this data is in: $DEST"
    read -r -p "PERMANENTLY delete $HOST:$HPC_DIR ? Type 'delete' to confirm: " ans
    [[ $ans == delete ]] || { log "step 5: skipped by user. Re-run the same command any time to delete later."; return 0; }
  fi

  emit_delete_job | upload "$STAGE.delete_job.sh" || { err "could not upload delete job"; return 1; }
  jid=$(submit "$STAGE.delete_job.sh" "$STAGE.delete_status" "$DELETE_SBATCH_OPTS" bundle_delete "$STAGE.delete" \
          "$HPC_DIR" "$STAGE" "$STAGE.delete_status") || { err "could not submit delete job"; return 1; }
  log "step 5: submitted delete job $jid; waiting"
  wait_job "$jid" "$STAGE.delete_status" delete || return 1
  rsh "rm -f $(q "$STAGE.delete_status") $(q "$STAGE.delete")_*.out; true" || true
  log "step 5: $HPC_DIR deleted on the HPC (and its temporary bundles in scratch)"
}

step6_cleanup() {
  local v="$WORKSPACE/verify_$NAME"
  if [[ -e $v ]]; then
    log "step 6: removing temporary verification files in workspace"
    chmod -R u+rwX "$v" 2>/dev/null || true
    rm -rf "$v"
  fi
}

# ------------------------------------------------------------------ main
preflight
if [[ -f $DEST/VERIFIED ]]; then
  log "dest already verified ($(cat "$DEST/VERIFIED")); skipping to step 5"
else
  step1_submit
  step2_wait
  step3_fetch
  step4_verify
fi
rc=0
step5_delete || rc=$?
step6_cleanup
if (( rc == 0 )); then log "ALL DONE"; else err "finished with problems in step 5 (your verified bundles in $DEST are safe; HPC data was not deleted)"; fi
exit $rc
