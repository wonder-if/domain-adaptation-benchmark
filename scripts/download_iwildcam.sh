#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${ROOT_DIR}/manifests/iwildcam_hf_sizes.tsv"

DEST=""
SOURCE="mirror"
PROXY_MODE="disable"
RETRY=50
RETRY_DELAY=5
CONNECT_TIMEOUT=30
STATUS_INTERVAL=2
JOBS=4
STATE_FILE_NAME=".iwildcam_downloaded.tsv"

if [[ -t 1 ]]; then
  C_RESET=$'\033[0m'
  C_DIM=$'\033[2m'
  C_BOLD=$'\033[1m'
  C_BLUE=$'\033[38;5;39m'
  C_MINT=$'\033[38;5;43m'
  C_GREEN=$'\033[38;5;78m'
  C_YELLOW=$'\033[38;5;220m'
  C_RED=$'\033[38;5;203m'
else
  C_RESET=""
  C_DIM=""
  C_BOLD=""
  C_BLUE=""
  C_MINT=""
  C_GREEN=""
  C_YELLOW=""
  C_RED=""
fi

print_line() {
  printf '%b\n' "$1"
}

print_meta() {
  local label="$1"
  local value="$2"
  print_line "  ${C_DIM}${label}:${C_RESET} ${value}"
}

print_header() {
  print_line "${C_BLUE}╭──────────────────────────────────────────────╮${C_RESET}"
  print_line "${C_BLUE}│${C_RESET} ${C_BOLD}iWildCam mirror downloader${C_RESET} ${C_DIM}(^･o･^)ﾉ\"${C_RESET}"
  print_line "${C_BLUE}╰──────────────────────────────────────────────╯${C_RESET}"
}

format_bytes() {
  numfmt --to=iec-i --suffix=B "$1"
}

format_duration() {
  local total="$1"
  local h=$((total / 3600))
  local m=$(((total % 3600) / 60))
  local s=$((total % 60))

  if (( h > 0 )); then
    printf '%02d:%02d:%02d' "${h}" "${m}" "${s}"
  else
    printf '%02d:%02d' "${m}" "${s}"
  fi
}

render_bar() {
  local done="$1"
  local total="$2"
  local width=24
  local units=0
  local full=0
  local partial=0
  local bar=""
  local i
  local partial_char=""

  if (( total > 0 )); then
    units=$((done * width * 8 / total))
  fi

  full=$((units / 8))
  partial=$((units % 8))

  case "${partial}" in
    0) partial_char="" ;;
    1) partial_char="▏" ;;
    2) partial_char="▎" ;;
    3) partial_char="▍" ;;
    4) partial_char="▌" ;;
    5) partial_char="▋" ;;
    6) partial_char="▊" ;;
    7) partial_char="▉" ;;
  esac

  for ((i = 0; i < width; i++)); do
    if (( i < full )); then
      bar+="█"
    elif (( i == full )) && [[ -n "${partial_char}" ]]; then
      bar+="${partial_char}"
    else
      bar+="·"
    fi
  done

  printf '%s' "${bar}"
}

truncate_name() {
  local value="$1"
  local limit="${2:-54}"

  if (( ${#value} <= limit )); then
    printf '%s' "${value}"
  else
    printf '%s…' "${value:0:$((limit - 1))}"
  fi
}

compact_name() {
  local relpath="$1"
  local base
  base="$(basename "${relpath}")"

  if [[ "${base}" =~ ^(train|test)-0*([0-9]+)-of-0*([0-9]+)\.parquet$ ]]; then
    printf '%s %s/%s' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" "${BASH_REMATCH[3]}"
    return 0
  fi

  case "${base}" in
    .gitattributes) printf 'git attrs' ;;
    README.md) printf 'readme' ;;
    iwildcam2020_megadetector_results.json) printf 'megadetector' ;;
    iwildcam2020_test_information.json) printf 'test meta' ;;
    iwildcam2020_train_annotations.json) printf 'train anno' ;;
    *) printf '%s' "${base}" ;;
  esac
}

state_file_path() {
  printf '%s/%s' "${DEST}" "${STATE_FILE_NAME}"
}

declare -A COMPLETED_FILES=()
declare -a ALL_RELPATHS=()
declare -a ALL_SIZES=()
declare -a PENDING_RELPATHS=()
declare -a PENDING_SIZES=()
declare -a ACTIVE_PIDS=()
declare -a ACTIVE_RELPATHS=()
declare -a ACTIVE_TARGETS=()
declare -a ACTIVE_SIZES=()
declare -a ACTIVE_LABELS=()
STATE_DIRTY=0

load_state() {
  local state_file="$1"
  local relpath=""
  local recorded_size=""

  COMPLETED_FILES=()
  [[ -f "${state_file}" ]] || return 0

  while IFS=$'\t' read -r relpath recorded_size; do
    [[ -n "${relpath}" ]] || continue
    COMPLETED_FILES["${relpath}"]="${recorded_size}"
  done < "${state_file}"
}

is_completed() {
  local relpath="$1"
  [[ -n "${COMPLETED_FILES[${relpath}]+x}" ]]
}

mark_completed() {
  local state_file="$1"
  local relpath="$2"
  local expected_size="$3"

  if is_completed "${relpath}"; then
    return 0
  fi

  COMPLETED_FILES["${relpath}"]="${expected_size}"
  printf '%s\t%s\n' "${relpath}" "${expected_size}" >> "${state_file}"
}

rewrite_state_file() {
  local state_file="$1"
  local relpath

  : > "${state_file}"
  for relpath in "${!COMPLETED_FILES[@]}"; do
    printf '%s\t%s\n' "${relpath}" "${COMPLETED_FILES[${relpath}]}" >> "${state_file}"
  done
  STATE_DIRTY=0
}

usage() {
  cat <<'EOF'
Usage:
  bash scripts/download_iwildcam.sh [options]

Options:
  --dest PATH             Target directory. Required.
  --source mirror|hf      Download source. "mirror" = hf-mirror.com, "hf" = huggingface.co
  --proxy keep|disable    Keep current proxy env or disable all proxy env vars
  --jobs N                Concurrent shard downloads. Default: 4
  --retry N               Curl retry count. Default: 50
  --retry-delay N         Delay between retries in seconds. Default: 5
  -h, --help              Show this help message

Examples:
  bash scripts/download_iwildcam.sh --dest /path/to/iwildcam
  bash scripts/download_iwildcam.sh --dest /path/to/iwildcam --jobs 6 --source mirror --proxy disable
  bash scripts/download_iwildcam.sh --dest /path/to/iwildcam --source hf --proxy keep
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest)
      DEST="$2"
      shift 2
      ;;
    --source)
      SOURCE="$2"
      shift 2
      ;;
    --proxy)
      PROXY_MODE="$2"
      shift 2
      ;;
    --jobs)
      JOBS="$2"
      shift 2
      ;;
    --retry)
      RETRY="$2"
      shift 2
      ;;
    --retry-delay)
      RETRY_DELAY="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${DEST}" ]]; then
  echo "Missing required argument: --dest PATH" >&2
  usage >&2
  exit 1
fi

case "${SOURCE}" in
  mirror)
    BASE_URL="https://hf-mirror.com/datasets/anngrosha/iWildCam2020/resolve/main"
    ;;
  hf)
    BASE_URL="https://huggingface.co/datasets/anngrosha/iWildCam2020/resolve/main"
    ;;
  *)
    echo "Invalid --source: ${SOURCE}. Use mirror or hf." >&2
    exit 1
    ;;
esac

case "${PROXY_MODE}" in
  disable)
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
    ;;
  keep)
    ;;
  *)
    echo "Invalid --proxy: ${PROXY_MODE}. Use keep or disable." >&2
    exit 1
    ;;
esac

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Manifest not found: ${MANIFEST}" >&2
  exit 1
fi

if ! [[ "${JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid --jobs: ${JOBS}. Use a positive integer." >&2
  exit 1
fi

mkdir -p "${DEST}"
STATE_FILE="$(state_file_path)"
touch "${STATE_FILE}"
load_state "${STATE_FILE}"

prepare_queue() {
  local relpath expected_size target actual_size

  while IFS=$'\t' read -r relpath expected_size; do
    [[ -n "${relpath}" ]] || continue
    ALL_RELPATHS+=("${relpath}")
    ALL_SIZES+=("${expected_size}")

    target="${DEST}/$(basename "${relpath}")"

    if is_completed "${relpath}"; then
      if [[ -f "${target}" ]]; then
        continue
      fi
      unset 'COMPLETED_FILES[$relpath]'
      STATE_DIRTY=1
    fi

    if [[ -f "${target}" ]]; then
      actual_size="$(stat -c %s "${target}")"
      if [[ "${actual_size}" == "${expected_size}" ]]; then
        mark_completed "${STATE_FILE}" "${relpath}" "${expected_size}"
        continue
      fi
      rm -f "${target}"
    fi

    PENDING_RELPATHS+=("${relpath}")
    PENDING_SIZES+=("${expected_size}")
  done < "${MANIFEST}"
}

total_bytes() {
  local sum=0
  local size
  for size in "${ALL_SIZES[@]}"; do
    sum=$((sum + size))
  done
  printf '%s' "${sum}"
}

completed_bytes() {
  local relpath sum=0
  for relpath in "${ALL_RELPATHS[@]}"; do
    if is_completed "${relpath}"; then
      sum=$((sum + COMPLETED_FILES[${relpath}]))
    fi
  done
  printf '%s' "${sum}"
}

active_bytes() {
  local i sum=0 size
  for i in "${!ACTIVE_PIDS[@]}"; do
    [[ -n "${ACTIVE_PIDS[i]:-}" ]] || continue
    if [[ -f "${ACTIVE_TARGETS[i]}" ]]; then
      size="$(stat -c %s "${ACTIVE_TARGETS[i]}" 2>/dev/null || printf '0')"
      sum=$((sum + size))
    fi
  done
  printf '%s' "${sum}"
}

active_labels() {
  local labels=()
  local i
  for i in "${!ACTIVE_PIDS[@]}"; do
    [[ -n "${ACTIVE_PIDS[i]:-}" ]] || continue
    labels+=("${ACTIVE_LABELS[i]}")
  done

  if (( ${#labels[@]} == 0 )); then
    printf 'waiting'
  elif (( ${#labels[@]} <= 3 )); then
    local joined=""
    local label
    for label in "${labels[@]}"; do
      if [[ -n "${joined}" ]]; then
        joined+=", "
      fi
      joined+="${label}"
    done
    printf '%s' "${joined}"
  else
    printf '%s, %s, %s +%s' "${labels[0]}" "${labels[1]}" "${labels[2]}" "$(( ${#labels[@]} - 3 ))"
  fi
}

launch_download() {
  local slot="$1"
  local relpath="$2"
  local expected_size="$3"
  local target="${DEST}/$(basename "${relpath}")"
  local url="${BASE_URL}/${relpath}?download=true"

  (
    curl -L \
      --silent \
      --show-error \
      --retry "${RETRY}" \
      --retry-delay "${RETRY_DELAY}" \
      --retry-all-errors \
      --connect-timeout "${CONNECT_TIMEOUT}" \
      --max-time 0 \
      -o "${target}" \
      "${url}"
  ) &

  ACTIVE_PIDS[slot]=$!
  ACTIVE_RELPATHS[slot]="${relpath}"
  ACTIVE_TARGETS[slot]="${target}"
  ACTIVE_SIZES[slot]="${expected_size}"
  ACTIVE_LABELS[slot]="$(compact_name "${relpath}")"
}

clear_slot() {
  local slot="$1"
  ACTIVE_PIDS[slot]=""
  ACTIVE_RELPATHS[slot]=""
  ACTIVE_TARGETS[slot]=""
  ACTIVE_SIZES[slot]=""
  ACTIVE_LABELS[slot]=""
}

process_finished_jobs() {
  local slot pid relpath target expected_size actual_size status

  for slot in "${!ACTIVE_PIDS[@]}"; do
    pid="${ACTIVE_PIDS[slot]:-}"
    [[ -n "${pid}" ]] || continue
    if kill -0 "${pid}" 2>/dev/null; then
      continue
    fi

    status=0
    if ! wait "${pid}"; then
      status=$?
    fi

    relpath="${ACTIVE_RELPATHS[slot]}"
    target="${ACTIVE_TARGETS[slot]}"
    expected_size="${ACTIVE_SIZES[slot]}"

    if (( status != 0 )); then
      print_line ""
      print_line "${C_RED}Download failed: $(compact_name "${relpath}")${C_RESET}" >&2
      print_line "${C_DIM}Target: ${target}${C_RESET}" >&2
      exit 1
    fi

    actual_size="$(stat -c %s "${target}" 2>/dev/null || printf '0')"
    if [[ "${actual_size}" != "${expected_size}" ]]; then
      print_line ""
      print_line "${C_RED}Size mismatch: $(compact_name "${relpath}") (${actual_size} != ${expected_size})${C_RESET}" >&2
      print_line "${C_DIM}The shard will stay unmarked and be retried on the next run.${C_RESET}" >&2
      exit 1
    fi

    mark_completed "${STATE_FILE}" "${relpath}" "${expected_size}"
    clear_slot "${slot}"
  done
}

render_status() {
  local total_bytes_value="$1"
  local completed_bytes_value="$2"
  local active_bytes_value="$3"
  local done_bytes now elapsed speed remaining eta percent bar summary cols line
  local files_done files_total

  done_bytes=$((completed_bytes_value + active_bytes_value))
  now="$(date +%s)"
  elapsed=$((now - START_TIME))
  speed=$(( (done_bytes - LAST_DONE_BYTES) / STATUS_INTERVAL ))
  if (( speed < 0 )); then
    speed=0
  fi

  remaining=$((total_bytes_value - done_bytes))
  if (( remaining < 0 )); then
    remaining=0
  fi

  if (( speed > 0 )); then
    eta=$(((remaining + speed - 1) / speed))
  else
    eta=0
  fi

  percent=0
  if (( total_bytes_value > 0 )); then
    percent=$((done_bytes * 100 / total_bytes_value))
  fi

  files_done="${#COMPLETED_FILES[@]}"
  files_total="${#ALL_RELPATHS[@]}"
  bar="$(render_bar "${done_bytes}" "${total_bytes_value}")"
  summary="$(active_labels)"
  cols="${COLUMNS:-}"
  if [[ -z "${cols}" ]] && command -v tput >/dev/null 2>&1; then
    cols="$(tput cols 2>/dev/null || printf '120')"
  fi
  [[ -n "${cols}" ]] || cols=120

  line="${C_BLUE}${bar}${C_RESET} ${percent}% ${C_BOLD}${files_done}/${files_total}${C_RESET}"
  line+=" ${C_DIM}| $(format_bytes "${done_bytes}")/$(format_bytes "${total_bytes_value}")${C_RESET}"
  line+=" ${C_DIM}| $(format_bytes "${speed}")/s${C_RESET}"
  line+=" ${C_DIM}| $(format_duration "${elapsed}")${C_RESET}"
  if (( speed > 0 )); then
    line+=" ${C_DIM}| eta $(format_duration "${eta}")${C_RESET}"
  else
    line+=" ${C_DIM}| eta --:--${C_RESET}"
  fi
  line+=" ${C_MINT}| now ${summary}${C_RESET}"

  if (( cols > 12 )); then
    line="$(truncate_name "${line}" "$((cols - 1))")"
  fi

  printf '\r\033[2K%b' "${line}"
  LAST_DONE_BYTES="${done_bytes}"
}

finish_status() {
  if [[ -t 1 ]]; then
    printf '\n'
  fi
}

prepare_queue
TOTAL_BYTES="$(total_bytes)"
START_TIME="$(date +%s)"
LAST_DONE_BYTES="$(completed_bytes)"

print_header
print_meta "dest" "${DEST}"
print_meta "source" "${SOURCE} -> ${BASE_URL}"
print_meta "jobs" "${JOBS}"
print_meta "ready" "${#COMPLETED_FILES[@]} / ${#ALL_RELPATHS[@]}"
print_meta "size" "$(format_bytes "${TOTAL_BYTES}")"
print_line ""

PENDING_INDEX=0
while (( PENDING_INDEX < ${#PENDING_RELPATHS[@]} )) || (( ${#ACTIVE_PIDS[@]} > 0 )); do
  while (( PENDING_INDEX < ${#PENDING_RELPATHS[@]} )); do
    if (( ${#ACTIVE_PIDS[@]} >= JOBS )); then
      break
    fi

    slot="${#ACTIVE_PIDS[@]}"
    launch_download "${slot}" "${PENDING_RELPATHS[PENDING_INDEX]}" "${PENDING_SIZES[PENDING_INDEX]}"
    PENDING_INDEX=$((PENDING_INDEX + 1))
  done

  process_finished_jobs

  new_active_pids=()
  new_active_relpaths=()
  new_active_targets=()
  new_active_sizes=()
  new_active_labels=()
  for i in "${!ACTIVE_PIDS[@]}"; do
    [[ -n "${ACTIVE_PIDS[i]:-}" ]] || continue
    new_active_pids+=("${ACTIVE_PIDS[i]}")
    new_active_relpaths+=("${ACTIVE_RELPATHS[i]}")
    new_active_targets+=("${ACTIVE_TARGETS[i]}")
    new_active_sizes+=("${ACTIVE_SIZES[i]}")
    new_active_labels+=("${ACTIVE_LABELS[i]}")
  done
  ACTIVE_PIDS=("${new_active_pids[@]}")
  ACTIVE_RELPATHS=("${new_active_relpaths[@]}")
  ACTIVE_TARGETS=("${new_active_targets[@]}")
  ACTIVE_SIZES=("${new_active_sizes[@]}")
  ACTIVE_LABELS=("${new_active_labels[@]}")

  render_status "${TOTAL_BYTES}" "$(completed_bytes)" "$(active_bytes)"

  if (( PENDING_INDEX < ${#PENDING_RELPATHS[@]} )) || (( ${#ACTIVE_PIDS[@]} > 0 )); then
    sleep "${STATUS_INTERVAL}"
  fi
done

if (( STATE_DIRTY )); then
  rewrite_state_file "${STATE_FILE}"
fi

finish_status
print_line "${C_GREEN}done${C_RESET} ${C_DIM}| all shards are ready${C_RESET}"
print_meta "dest" "${DEST}"
print_meta "state" "${STATE_FILE}"
