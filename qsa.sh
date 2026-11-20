#!/usr/bin/env bash
# One-stop local deployment of the pinned QSA build. No root is needed except `deps`.
set -Eeuo pipefail

readonly QSA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly UPSTREAM_REPO=13thblade/ninfer-all-QSA
readonly UPSTREAM_COMMIT=29c79545429f03c2ec9d74370b3b2c59b62cca0d
readonly UPSTREAM_TAG=win-cuda13.4-29c79545429f
readonly BUILD_REPO=cnnlei/ninfer-qsa-ubuntu-build
readonly BUILD_NAME="ninfer-linux-x64-cuda13.4-sm120a-${UPSTREAM_COMMIT:0:12}"
readonly GGUF_NAME=Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf
readonly MMPROJ_NAME=mmproj-Qwen3.8-27B-BF16.gguf
cd "$QSA_ROOT"

die() { printf '错误: %s\n' "$*" >&2; exit 1; }
notice() { printf '[QSA] %s\n' "$*"; }
need() { command -v "$1" >/dev/null 2>&1 || die "缺少 $1；先运行 bash qsa.sh deps"; }
usage() {
  printf '%s\n' \
    'NInfer QSA · Ubuntu x86_64 / RTX 50' \
    '  bash qsa.sh init             创建自己的 qsa.env，不覆盖已有配置' \
    '  bash qsa.sh deps             显式安装系统依赖（需要 sudo）' \
    '  bash qsa.sh doctor           检查系统、GPU、运行包和模型' \
    '  bash qsa.sh install          下载 GitHub Release、校验、解压' \
    '  bash qsa.sh install --offline  使用 downloads/ 中的校验文件和分卷' \
    '  bash qsa.sh prepare          独立转换环境 + HF v2 索引器 + 小型元数据' \
    '  bash qsa.sh convert [--check]  自动下载 ISTA IQ3_S MTP → QSA v2 .ninfer' \
    '  bash qsa.sh all              install → convert → start' \
    '  bash qsa.sh plan             预览启动参数（不运行）' \
    '  bash qsa.sh start|stop|restart|status' \
    '  bash qsa.sh logs [-f]        查看/实时跟踪日志' \
    '  bash qsa.sh test             发送本机最小聊天请求' \
    '  bash qsa.sh stop --force     超时后才强制结束本脚本启动的进程'
}

readonly CONFIG="${QSA_CONFIG:-$QSA_ROOT/qsa.env}"
if [[ -f "$CONFIG" ]]; then
  # The file is a trusted local Bash config, not downloaded code; git ignores it.
  # shellcheck disable=SC1090
  source "$CONFIG"
fi

: "${QSA_ASSETS_DIR:=$QSA_ROOT/assets}"
: "${QSA_GGUF:=$QSA_ASSETS_DIR/inputs/$GGUF_NAME}"
: "${QSA_MMPROJ:=$QSA_ASSETS_DIR/inputs/$MMPROJ_NAME}"
: "${QSA_OUTPUT_MODEL:=$QSA_ROOT/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-QSA-v2-mtp.ninfer}"
: "${QSA_MODEL:=$QSA_OUTPUT_MODEL}"
: "${QSA_BASE_MODEL_DIR:=$QSA_ROOT/assets/base-model}"
: "${QSA_DONOR_GGUF:=}"
: "${QSA_SOURCE_DIR:=$QSA_ROOT/upstream}"
: "${QSA_RUNTIME_ROOT:=$QSA_ROOT/runtime}"
: "${QSA_DOWNLOAD_DIR:=$QSA_ROOT/downloads}"
: "${QSA_VENV:=$QSA_ROOT/.venv}"
: "${QSA_STATE_DIR:=$QSA_ROOT/run}"
: "${QSA_SERVE_BIN:=$QSA_RUNTIME_ROOT/$BUILD_NAME/ninfer-serve}"
: "${QSA_RELEASE_TAG:=}"
: "${QSA_MODEL_ID:=qwen3.8-27b-qsa-v2-iq3s}"
: "${QSA_BIND:=127.0.0.1}"
: "${QSA_PORT:=18201}"
: "${QSA_GPU:=0}"
: "${QSA_API_KEY_FILE:=}"
: "${QSA_CONTEXT:=262144}"
: "${QSA_TOP_BLOCKS:=1024}"
: "${QSA_KV_RESIDENT:=8192}"
: "${QSA_KV_DTYPE:=int8}"
: "${QSA_HOST_CACHE_MIB:=16384}"
: "${QSA_HEADROOM_MIB:=725}"
: "${QSA_PREFILL_CHUNK:=512}"
: "${QSA_MAX_CONCURRENCY:=1}"
: "${QSA_MAX_OUTPUT:=32768}"
: "${QSA_EMBEDDING_RESIDENCY:=host}"
: "${QSA_REASONING_EFFORT:=xhigh}"
: "${QSA_SPARSE_PREFILL:=0}"
: "${QSA_GDN_FP16:=0}"
: "${QSA_MTP:=1}"
: "${QSA_DRAFT_TOKENS:=3}"
: "${QSA_NGRAM_DRAFT_TOKENS:=15}"
: "${QSA_VISION:=1}"
: "${QSA_VISION_MAX_MERGED:=512}"
: "${QSA_PROPOSAL_HEAD:=0}"
: "${QSA_START_WAIT:=60}"
: "${QSA_STOP_WAIT:=60}"
: "${QSA_PYTHON:=}"

# Normalize paths before a subprocess changes working directory into the converter source.
for path_var in QSA_GGUF QSA_MMPROJ QSA_OUTPUT_MODEL QSA_MODEL QSA_BASE_MODEL_DIR QSA_DONOR_GGUF \
    QSA_SOURCE_DIR QSA_RUNTIME_ROOT QSA_DOWNLOAD_DIR QSA_ASSETS_DIR QSA_VENV QSA_STATE_DIR \
    QSA_SERVE_BIN QSA_API_KEY_FILE; do
  if [[ -n "${!path_var}" ]]; then printf -v "$path_var" '%s' "$(realpath -m -- "${!path_var}")"; fi
done

readonly PID_RECORD="$QSA_STATE_DIR/server.pid"
readonly LOG_FILE="$QSA_STATE_DIR/server.log"
readonly VENV_PYTHON="$QSA_VENV/bin/python"
readonly RAW_INDEXER="$QSA_ASSETS_DIR/hf-v2/qsa-indexers.safetensors"
readonly INDEXER="$QSA_ASSETS_DIR/hf-v2/indexer_tensors-v2-ninfer.safetensors"
readonly PREPARED_BASE="$QSA_ASSETS_DIR/base-model-qsa-v2"

require_config() { [[ -f "$CONFIG" ]] || die '先运行 bash qsa.sh init 并编辑 qsa.env'; }
require_linux() {
  [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die '仅支持 Linux x86_64'
}
uint_range() {
  local name=$1 value=$2 min=$3 max=$4
  [[ "$value" =~ ^(0|[1-9][0-9]{0,8})$ ]] || die "$name 必须是整数"
  (( value >= min && value <= max )) || die "$name 必须在 $min..$max"
}
flag01() { [[ "$2" == 0 || "$2" == 1 ]] || die "$1 必须是 0 或 1"; }

init_config() {
  if [[ -e "$CONFIG" ]]; then notice "已有配置：$CONFIG"; return; fi
  mkdir -p -- "$(dirname -- "$CONFIG")"
  (umask 077; set -o noclobber; cp --no-clobber -- "$QSA_ROOT/qsa.env.example" "$CONFIG")
  chmod 600 -- "$CONFIG"
  notice "已生成 $CONFIG；默认自动下载指定 ISTA IQ3_S MTP / BF16 mmproj，也可填写已有文件路径。"
}

deps() {
  require_linux
  need apt-get
  local -a elevate=()
  (( EUID == 0 )) || { need sudo; elevate=(sudo); }
  local -a packages=(git curl ca-certificates python3 python3-venv tar gzip coreutils util-linux libstdc++6 zlib1g)
  local list="$QSA_RUNTIME_ROOT/$BUILD_NAME/packages/multi.txt" item
  if [[ -f "$list" ]]; then
    while IFS= read -r item; do
      [[ -n "$item" ]] || continue
      [[ "$item" =~ ^[a-z0-9][a-z0-9+.-]*(:amd64)?$ ]] || die "运行库清单条目无效: $item"
      packages+=("$item")
    done < "$list"
  else
    packages+=(libcurl4t64)
  fi
  "${elevate[@]}" apt-get update
  "${elevate[@]}" apt-get install -y --no-install-recommends "${packages[@]}"
}

lock_state() {
  need flock
  mkdir -p -- "$QSA_STATE_DIR"
  chmod 700 -- "$QSA_STATE_DIR"
  exec 9>"$QSA_STATE_DIR/control.lock"
  flock -n 9 || die '另一个 QSA 操作正在运行，请稍后再试'
}

install_runtime() {
  require_linux; need python3; need curl
  local -a args=(--repo "$BUILD_REPO" --commit "$UPSTREAM_COMMIT"
      --runtime-root "$QSA_RUNTIME_ROOT" --downloads "$QSA_DOWNLOAD_DIR")
  [[ -z "$QSA_RELEASE_TAG" ]] || args+=(--tag "$QSA_RELEASE_TAG")
  if [[ "${1:-}" == --offline ]]; then args+=(--offline); shift; fi
  (( $# == 0 )) || die 'install 只支持可选参数 --offline'
  python3 "$QSA_ROOT/scripts/install_runtime.py" "${args[@]}"
}

prepare_converter() {
  need git; need python3; need curl; need sha256sum
  if [[ ! -e "$QSA_SOURCE_DIR" ]]; then
    git clone --depth 1 --branch "$UPSTREAM_TAG" "https://github.com/$UPSTREAM_REPO.git" "$QSA_SOURCE_DIR"
  fi
  [[ "$(git -C "$QSA_SOURCE_DIR" rev-parse HEAD 2>/dev/null)" == "$UPSTREAM_COMMIT" ]] ||
    die "$QSA_SOURCE_DIR 不是固定的 QSA 源码版本；不会覆盖它"
  [[ -z "$(git -C "$QSA_SOURCE_DIR" status --porcelain --untracked-files=no)" ]] ||
    die 'QSA 转换源码有未提交修改；请使用独立目录保存自定义修改'
  if [[ ! -x "$VENV_PYTHON" ]]; then
    local python_bin=python3
    if [[ -n "$QSA_PYTHON" ]]; then python_bin=$QSA_PYTHON
    elif command -v python3.11 >/dev/null; then python_bin=python3.11; fi
    need "$python_bin"
    "$python_bin" -m venv "$QSA_VENV"
  fi
  "$VENV_PYTHON" -c 'import sys; assert sys.version_info >= (3,11)' || die '转换环境需要 Python 3.11 或更新版本'
  if ! "$VENV_PYTHON" -c 'import torch, numpy, gguf, safetensors, huggingface_hub, jinja2' >/dev/null 2>&1; then
    notice '安装 CPU 版 PyTorch 和转换依赖到独立虚拟环境（第一次需要下载）'
    "$VENV_PYTHON" -m pip install --disable-pip-version-check torch --index-url https://download.pytorch.org/whl/cpu
    "$VENV_PYTHON" -m pip install --disable-pip-version-check numpy gguf safetensors huggingface_hub jinja2
  fi
  "$VENV_PYTHON" -c 'import torch, numpy, gguf, safetensors, huggingface_hub, jinja2; print("Python converter dependencies: OK")'
  # Do NOT use the GitHub release's older indexer or external GDN tensor assets.
  "$VENV_PYTHON" "$QSA_ROOT/scripts/hf_v2.py" indexer --source "$RAW_INDEXER" \
    --output "$INDEXER" --upstream "$QSA_SOURCE_DIR"
  local -a metadata_args=("$QSA_BASE_MODEL_DIR" --prepared-dir "$PREPARED_BASE")
  [[ "$QSA_VISION" != 1 ]] || metadata_args+=(--vision)
  "$VENV_PYTHON" "$QSA_ROOT/scripts/fetch_base_metadata.py" "${metadata_args[@]}"
  notice '转换环境和 Hugging Face v2 索引器已准备并通过上游读取器校验。'
}

convert_model() {
  require_config
  flag01 QSA_VISION "$QSA_VISION"; flag01 QSA_MTP "$QSA_MTP"; flag01 QSA_PROPOSAL_HEAD "$QSA_PROPOSAL_HEAD"
  local check=0
  if [[ "${1:-}" == --check ]]; then check=1; shift; fi
  (( $# == 0 )) || die 'convert 只支持可选参数 --check'
  [[ -z "$QSA_DONOR_GGUF" ]] || die '此版固定使用 HF v2 索引器，请从 qsa.env 移除旧 QSA_DONOR_GGUF 设置'
  if (( ! check )) && [[ -e "$QSA_OUTPUT_MODEL" ]]; then
    [[ -f "$QSA_OUTPUT_MODEL.conversion.json" && -f "$QSA_OUTPUT_MODEL.qsa-v2.json" && -s "$QSA_OUTPUT_MODEL" ]] || die '输出已存在但缺少 v2 来源证明，拒绝覆盖或误用 v1；请更换 QSA_OUTPUT_MODEL'
    local -a verify_args=(--model "$QSA_OUTPUT_MODEL")
    [[ "$QSA_MTP" != 1 ]] || verify_args+=(--mtp)
    [[ "$QSA_VISION" != 1 ]] || verify_args+=(--vision)
    python3 "$QSA_ROOT/scripts/hf_v2.py" verify-marker "${verify_args[@]}"
    notice "已有已校验的 v2 模型：$QSA_OUTPUT_MODEL（不重复转换或覆盖）"
    return
  fi
  [[ "$QSA_OUTPUT_MODEL" == *.ninfer ]] || die 'QSA_OUTPUT_MODEL 必须是 .ninfer 路径'
  [[ ( ! -e "$QSA_OUTPUT_MODEL.conversion.json" && ! -e "$QSA_OUTPUT_MODEL.qsa-v2.json" ) || $check == 1 ]] || die '同名转换报告已存在；请更换 QSA_OUTPUT_MODEL'
  prepare_converter
  local -a input_args=(--gguf "$QSA_GGUF")
  [[ "$QSA_VISION" != 1 ]] || input_args+=(--mmproj "$QSA_MMPROJ")
  "$VENV_PYTHON" "$QSA_ROOT/scripts/hf_v2.py" model-inputs "${input_args[@]}"
  local components=text target="$QSA_OUTPUT_MODEL"
  [[ "$QSA_VISION" != 1 ]] || components+=,vision
  [[ "$QSA_MTP" != 1 ]] || components+=,mtp
  if (( ! check )); then
    mkdir -p -- "$(dirname -- "$QSA_OUTPUT_MODEL")"
    target="${QSA_OUTPUT_MODEL%.ninfer}.partial.$$.ninfer"
    [[ ! -e "$target" ]] || die "临时输出已存在：$target"
    notice "CPU 转换中：$QSA_GGUF -> $QSA_OUTPUT_MODEL"
  fi
  local -a args=(--base-model "$(realpath -- "$PREPARED_BASE")" --gguf "$(realpath -- "$QSA_GGUF")"
      --components "$components" --out "$(realpath -m -- "$target")" --device cpu --name "$QSA_MODEL_ID")
  if [[ "$QSA_VISION" == 1 ]]; then args+=(--vision "$(realpath -- "$QSA_MMPROJ")"); fi
  if [[ "$QSA_PROPOSAL_HEAD" == 0 ]]; then args+=(--no-proposal); else args+=(--proposal); fi
  (( ! check )) || args+=(--check-only)
  local indexer_path
  indexer_path=$(realpath -- "$INDEXER")
  if ! (cd "$QSA_SOURCE_DIR" && env -u QSA_INDEXER_GGUF NINFER_QSA_INDEXER_ASSET="$indexer_path" \
        NINFER_GDN_CONTROLS_BF16=off "$VENV_PYTHON" tools/convert/convert_qsa27b.py "${args[@]}"); then
    die "模型转换失败，原模型未被改动。若生成了临时输出，请检查 $target"
  fi
  if (( check )); then return; fi
  [[ -s "$target" && -s "$target.conversion.json" ]] || die '转换器未生成完整模型及报告'
  local -a record_args=(--model "$target" --upstream "$QSA_SOURCE_DIR")
  [[ "$QSA_MTP" != 1 ]] || record_args+=(--mtp)
  [[ "$QSA_VISION" != 1 ]] || record_args+=(--vision)
  "$VENV_PYTHON" "$QSA_ROOT/scripts/hf_v2.py" record-model "${record_args[@]}"
  [[ ! -e "$QSA_OUTPUT_MODEL" && ! -e "$QSA_OUTPUT_MODEL.conversion.json" && ! -e "$QSA_OUTPUT_MODEL.qsa-v2.json" ]] || die '目标路径在转换期间已出现，拒绝覆盖'
  mv --no-clobber -- "$target.conversion.json" "$QSA_OUTPUT_MODEL.conversion.json"
  [[ ! -e "$target.conversion.json" ]] || die '转换报告移至目标位置失败'
  mv --no-clobber -- "$target.qsa-v2.json" "$QSA_OUTPUT_MODEL.qsa-v2.json"
  [[ ! -e "$target.qsa-v2.json" ]] || die 'v2 来源证明移至目标位置失败'
  mv --no-clobber -- "$target" "$QSA_OUTPUT_MODEL"
  [[ -s "$QSA_OUTPUT_MODEL" && ! -e "$target" ]] || die '模型移至目标位置失败'
  notice "转换完成：$QSA_OUTPUT_MODEL"
}

validate_config() {
  uint_range QSA_PORT "$QSA_PORT" 1024 65535
  uint_range QSA_GPU "$QSA_GPU" 0 64
  uint_range QSA_CONTEXT "$QSA_CONTEXT" 4096 1048576
  uint_range QSA_TOP_BLOCKS "$QSA_TOP_BLOCKS" 512 65536
  if [[ "$QSA_KV_RESIDENT" != all ]]; then
    uint_range QSA_KV_RESIDENT "$QSA_KV_RESIDENT" 1 1048576
    (( QSA_KV_RESIDENT >= QSA_TOP_BLOCKS * 8 )) || die '此固定版本的 stream pool 下限是两个注意力窗口，请令 QSA_KV_RESIDENT >= QSA_TOP_BLOCKS×8'
    (( QSA_KV_RESIDENT < QSA_CONTEXT )) || die 'KV resident >= context 会关闭 streaming；如需全部驻显存，请显式设置 QSA_KV_RESIDENT=all'
  fi
  uint_range QSA_HOST_CACHE_MIB "$QSA_HOST_CACHE_MIB" 0 1048576
  uint_range QSA_HEADROOM_MIB "$QSA_HEADROOM_MIB" 0 16384
  uint_range QSA_PREFILL_CHUNK "$QSA_PREFILL_CHUNK" 128 4096
  (( QSA_PREFILL_CHUNK % 128 == 0 )) || die 'QSA_PREFILL_CHUNK 必须是 128 的倍数'
  uint_range QSA_MAX_CONCURRENCY "$QSA_MAX_CONCURRENCY" 1 8
  uint_range QSA_MAX_OUTPUT "$QSA_MAX_OUTPUT" 1 "$QSA_CONTEXT"
  uint_range QSA_DRAFT_TOKENS "$QSA_DRAFT_TOKENS" 1 15
  uint_range QSA_NGRAM_DRAFT_TOKENS "$QSA_NGRAM_DRAFT_TOKENS" 0 63
  if (( QSA_MAX_CONCURRENCY > 1 && QSA_NGRAM_DRAFT_TOKENS > 15 )); then die '并发 >1 时 ngram 草稿不能超过 15'; fi
  uint_range QSA_VISION_MAX_MERGED "$QSA_VISION_MAX_MERGED" 64 16384
  uint_range QSA_START_WAIT "$QSA_START_WAIT" 1 600
  uint_range QSA_STOP_WAIT "$QSA_STOP_WAIT" 1 600
  for name in QSA_SPARSE_PREFILL QSA_GDN_FP16 QSA_MTP QSA_VISION QSA_PROPOSAL_HEAD; do flag01 "$name" "${!name}"; done
  case "$QSA_KV_DTYPE" in int8|bf16|rk8v4|rk4v4|rk4v4-e8|rk2v4-e8|k8v4|nvfp4|fp8|bf16p) ;; *) die '不支持的 KV 精度' ;; esac
  case "$QSA_EMBEDDING_RESIDENCY" in device|host) ;; *) die 'Embedding 可选 device 或 host' ;; esac
  case "$QSA_REASONING_EFFORT" in none|minimal|low|medium|high|xhigh|max) ;; *) die '不支持的 reasoning effort' ;; esac
  [[ -n "$QSA_MODEL_ID" ]] || die 'QSA_MODEL_ID 不能为空'
  [[ -n "$QSA_BIND" ]] || die 'QSA_BIND 不能为空'
  if [[ "$QSA_BIND" != 127.0.0.1 && "$QSA_BIND" != localhost && "$QSA_BIND" != ::1 ]]; then
    [[ -n "$QSA_API_KEY_FILE" ]] || die '局域网监听必须配置 QSA_API_KEY_FILE，避免无鉴权暴露 API'
  fi
  if [[ -n "$QSA_API_KEY_FILE" ]]; then
    [[ -s "$QSA_API_KEY_FILE" && -f "$QSA_API_KEY_FILE" ]] || die 'API key 文件缺失或为空'
    local mode
    mode=$(stat -c %a -- "$QSA_API_KEY_FILE")
    (( (8#$mode & 077) == 0 )) || die 'API key 文件权限过宽，请 chmod 600'
  fi
}

api_url() {
  local host=$QSA_BIND
  [[ "$host" != 0.0.0.0 && "$host" != localhost ]] || host=127.0.0.1
  if [[ "$host" == :: || "$host" == '[::]' ]]; then host='[::1]'; fi
  if [[ "$host" == *:* && "$host" != \[*\] ]]; then host="[$host]"; fi
  printf 'http://%s:%s' "$host" "$QSA_PORT"
}

make_args() {
  local show_secret=${1:-0}
  validate_config
  SERVE_ARGS=("$QSA_SERVE_BIN" "$QSA_MODEL" --model-id "$QSA_MODEL_ID" --device "$QSA_GPU"
    --host "$QSA_BIND" --port "$QSA_PORT" --max-context "$QSA_CONTEXT"
    --qwen38-27b-qsa --qsa-top-blocks "$QSA_TOP_BLOCKS" --kv-resident "$QSA_KV_RESIDENT"
    --kv-dtype "$QSA_KV_DTYPE"
    --use-alt-prefix-caching --host-cache-mib "$QSA_HOST_CACHE_MIB"
    --prefill-chunk "$QSA_PREFILL_CHUNK" --max-concurrency "$QSA_MAX_CONCURRENCY"
    --default-max-tokens "$QSA_MAX_OUTPUT" --embedding-residency "$QSA_EMBEDDING_RESIDENCY"
    --reasoning-effort "$QSA_REASONING_EFFORT")
  # The pinned fork uses an explicit logical capacity for the streamed tier, not the
  # Device-resident Hybrid cache's free-VRAM auto allocator.
  if [[ "$QSA_KV_RESIDENT" == all ]]; then
    SERVE_ARGS+=(--kv-capacity auto --kv-headroom-mib "$QSA_HEADROOM_MIB")
  else
    SERVE_ARGS+=(--kv-capacity "$QSA_CONTEXT")
  fi
  if (( QSA_CONTEXT > 262144 )); then SERVE_ARGS+=(--rope-yarn); fi
  if [[ "$QSA_SPARSE_PREFILL" == 1 ]]; then SERVE_ARGS+=(--qsa-prefill); fi
  if [[ "$QSA_GDN_FP16" == 1 ]]; then SERVE_ARGS+=(--gdn-state-fp16); fi
  if [[ "$QSA_MTP" == 1 ]]; then
    SERVE_ARGS+=(--spec mtp --draft-tokens "$QSA_DRAFT_TOKENS" --ngram-draft-tokens "$QSA_NGRAM_DRAFT_TOKENS")
  else
    SERVE_ARGS+=(--spec none --ngram-draft-tokens 0)
  fi
  if [[ "$QSA_VISION" == 1 ]]; then SERVE_ARGS+=(--vision --vision-residency cpu --vision-max-merged "$QSA_VISION_MAX_MERGED"); fi
  if [[ -n "$QSA_API_KEY_FILE" ]]; then
    if (( show_secret )); then
      local key
      key=$(<"$QSA_API_KEY_FILE")
      [[ -n "$key" && "$key" != *$'\n'* && "$key" != *$'\r'* ]] || die 'API key 必须为单行非空文本'
      SERVE_ARGS+=(--api-key "$key")
    else
      SERVE_ARGS+=(--api-key '<hidden>')
    fi
  fi
}

# The PID record includes Linux boot id and /proc start tick to prevent killing a reused PID.
process_tick() {
  local pid=$1 stat tail
  [[ -r "/proc/$pid/stat" ]] || return 1
  IFS= read -r stat < "/proc/$pid/stat" || return 1
  tail=${stat##*) }
  local -a fields
  read -r -a fields <<< "$tail"
  [[ "${fields[0]:-}" != Z && "${fields[19]:-}" =~ ^[0-9]+$ ]] || return 1
  printf '%s\n' "${fields[19]}"
}

running() {
  local pid tick boot extra now
  [[ -f "$PID_RECORD" ]] || return 1
  read -r pid tick boot extra < "$PID_RECORD" || return 1
  [[ "$pid" =~ ^[1-9][0-9]*$ && "$tick" =~ ^[0-9]+$ && -z "${extra:-}" ]] || return 1
  [[ "$boot" == "$(</proc/sys/kernel/random/boot_id)" ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  now=$(process_tick "$pid") || return 1
  [[ "$now" == "$tick" ]] || return 1
  SERVER_PID=$pid
}

health() {
  curl --silent --show-error --fail --max-time 2 "$(api_url)/health" >/dev/null 2>&1
}

check_gpu() {
  command -v nvidia-smi >/dev/null || die '缺少 nvidia-smi：先安装兼容的 NVIDIA 显卡驱动'
  local specs
  specs=$(nvidia-smi -i "$QSA_GPU" --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader,nounits) || die 'NVIDIA 驱动不可用，请检查 nvidia-smi'
  printf '%s\n' "$specs"
  local capability driver
  capability=$(awk -F, '{gsub(/ /,"",$4); print $4}' <<< "$specs")
  driver=$(awk -F, '{gsub(/ /,"",$3); print $3}' <<< "$specs")
  [[ "$capability" == 12.0 ]] || die '此运行包只编译了 SM120a（GeForce RTX 50/消费级 Blackwell）'
  [[ "${driver%%.*}" =~ ^[0-9]+$ ]] && (( ${driver%%.*} >= 580 )) || die 'CUDA 13.x 需要兼容的新 NVIDIA 驱动（至少 R580 系列）'
}

check_runtime() {
  [[ -x "$QSA_SERVE_BIN" ]] || die '运行包不存在；先运行 bash qsa.sh install'
  local lib="$QSA_RUNTIME_ROOT/$BUILD_NAME/backends/cuda13/libninfer-runtime.so" output
  if [[ -f "$lib" ]]; then
    output=$(ldd "$lib" 2>&1) || die "无法检查运行库：$output"
    if [[ "$output" == *'not found'* ]]; then printf '%s\n' "$output" >&2; die '运行库缺失；运行 bash qsa.sh deps，并检查 NVIDIA 驱动'; fi
  fi
  "$QSA_SERVE_BIN" --version
  local help
  help=$("$QSA_SERVE_BIN" --help 2>&1) || die '无法读取 ninfer-serve 参数'
  [[ "$help" == *--qwen38-27b-qsa* && "$help" == *--kv-resident* ]] || die '这个运行程序不是 QSA 分支'
}

start_server() {
  require_config; require_linux; need curl
  if running; then notice "已经运行 (PID $SERVER_PID)。使用 restart 应用新配置"; return; fi
  [[ -f "$QSA_MODEL" && -s "$QSA_MODEL" ]] || die "模型不存在：$QSA_MODEL，先运行 convert"
  if [[ "$QSA_MODEL" == "$QSA_OUTPUT_MODEL" ]]; then
    [[ -f "$QSA_MODEL.qsa-v2.json" ]] || die '缺少 v2 来源证明；请先用 convert 生成正确模型'
    local -a marker_args=(--model "$QSA_MODEL" --quick)
    [[ "$QSA_MTP" != 1 ]] || marker_args+=(--mtp)
    [[ "$QSA_VISION" != 1 ]] || marker_args+=(--vision)
    python3 "$QSA_ROOT/scripts/hf_v2.py" verify-marker "${marker_args[@]}"
  fi
  make_args 1
  check_gpu
  check_runtime
  if health; then die "$(api_url) 已有服务，拒绝重复启动"; fi
  if command -v ss >/dev/null && [[ -n "$(ss -H -ltn "sport = :$QSA_PORT" 2>/dev/null)" ]]; then
    die "端口 $QSA_PORT 已被占用，修改 QSA_PORT 后重试"
  fi
  local available
  available=$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)
  if [[ "$available" =~ ^[0-9]+$ ]] && (( QSA_HOST_CACHE_MIB > available - 4096 )); then
    die "可用内存约 ${available}MiB，不能预留 $QSA_HOST_CACHE_MIB MiB Host Cache；请降低预算或释放内存"
  fi
  mkdir -p -- "$QSA_STATE_DIR"
  (umask 077; touch -- "$LOG_FILE")
  chmod 600 -- "$LOG_FILE"
  if [[ -e "$PID_RECORD" ]]; then rm -f -- "$PID_RECORD"; fi
  notice "正在启动 $QSA_MODEL_ID，日志：$LOG_FILE"
  # The child closes the control lock, so status/stop can act while the model loads.
  nohup "${SERVE_ARGS[@]}" >> "$LOG_FILE" 2>&1 < /dev/null 9>&- &
  local pid=$! tick boot
  tick=$(process_tick "$pid") || { tail -n 30 "$LOG_FILE" >&2; die '进程启动失败'; }
  boot=$(</proc/sys/kernel/random/boot_id)
  (umask 077; printf '%s %s %s\n' "$pid" "$tick" "$boot" > "$PID_RECORD.tmp")
  mv -- "$PID_RECORD.tmp" "$PID_RECORD"
  # Publish ownership before waiting so stop can be called during a slow model load.
  flock -u 9
  local deadline=$((SECONDS + QSA_START_WAIT))
  while (( SECONDS < deadline )); do
    running || { tail -n 40 "$LOG_FILE" >&2; die '服务提前退出'; }
    if health; then notice "服务已就绪：$(api_url) / API $(api_url)/v1 (PID $pid)"; return; fi
    sleep 1
  done
  notice "进程仍在加载 (PID $pid)，请用 status 或 logs -f 查看。"
}

stop_server() {
  local force=0
  if [[ "${1:-}" == --force ]]; then force=1; shift; fi
  (( $# == 0 )) || die 'stop 只支持可选参数 --force'
  if ! running; then
    notice '本工具管理的服务未运行。'
    [[ ! -e "$PID_RECORD" ]] || rm -f -- "$PID_RECORD"
    return
  fi
  local pid=$SERVER_PID deadline=$((SECONDS + QSA_STOP_WAIT))
  notice "正常停止 PID $pid (SIGTERM)"
  kill -TERM "$pid" 2>/dev/null || true
  while running && (( SECONDS < deadline )); do sleep 1; done
  if running; then
    if (( ! force )); then die '服务仍在退出，稍后再试 stop；如确需强制结束，使用 stop --force'; fi
    notice "正常停止超时，强制结束本工具记录的 PID $SERVER_PID"
    kill -KILL "$SERVER_PID"
    local kill_deadline=$((SECONDS + 5))
    while running && (( SECONDS < kill_deadline )); do sleep 1; done
    if running; then die '进程仍未退出，请检查系统'; fi
  fi
  rm -f -- "$PID_RECORD"
  notice '服务已停止。'
}

status_server() {
  need curl
  if running; then
    if health; then notice "运行中 / Ready (PID $SERVER_PID)"; else notice "运行中 / 正在加载或未就绪 (PID $SERVER_PID)"; fi
    printf 'API: %s/v1\n日志: %s\n' "$(api_url)" "$LOG_FILE"
  else
    notice '未运行'
  fi
}

doctor() {
  require_linux
  printf 'OS: '; if [[ -r /etc/os-release ]]; then awk -F= '/^PRETTY_NAME=/ {print $2}' /etc/os-release; fi
  getconf GNU_LIBC_VERSION || true
  printf '\nGPU (build target 120a):\n'; if command -v nvidia-smi >/dev/null; then nvidia-smi || true; else printf 'nvidia-smi 不存在。\n'; fi
  printf '\nMemory:\n'; free -h || true
  printf 'Memlock: '; ulimit -l || true
  printf '\nDisk:\n'; df -h "$QSA_ROOT" || true
  printf '\nRuntime: %s\n' "$QSA_SERVE_BIN"
  if [[ -x "$QSA_SERVE_BIN" ]]; then "$QSA_SERVE_BIN" --version || true; else printf '尚未安装；运行 install。\n'; fi
  printf 'GGUF: %s (%s)\n' "$QSA_GGUF" "$([[ -f "$QSA_GGUF" ]] && printf OK || printf missing)"
  printf 'Model: %s (%s)\n' "$QSA_MODEL" "$([[ -s "$QSA_MODEL" ]] && printf OK || printf missing)"
  printf 'Converter source: %s\n' "$QSA_SOURCE_DIR"
}

test_api() {
  require_config; need curl; need python3
  health || die '服务未就绪，请先 start/status'
  local body key
  local -a headers=(-H 'Content-Type: application/json')
  if [[ -n "$QSA_API_KEY_FILE" ]]; then key=$(<"$QSA_API_KEY_FILE"); headers+=(-H "Authorization: Bearer $key"); fi
  body=$(python3 -c 'import json,sys; print(json.dumps({"model":sys.argv[1],"messages":[{"role":"user","content":"1+1等于几？只回答数字。"}],"temperature":0,"max_tokens":64,"reasoning_effort":"none","stream":False},ensure_ascii=False))' "$QSA_MODEL_ID")
  curl --fail-with-body --silent --show-error --max-time 120 "${headers[@]}" --data "$body" "$(api_url)/v1/chat/completions"
  printf '\n'
}

cmd=${1:-help}; if (( $# > 0 )); then shift; fi
case "$cmd" in
  -h|--help|help) usage ;;
  init) init_config ;;
  deps) deps ;;
  doctor) doctor ;;
  install) lock_state; install_runtime "$@" ;;
  prepare) lock_state; prepare_converter ;;
  convert) lock_state; convert_model "$@" ;;
  plan) require_config; make_args 0; printf '%q ' "${SERVE_ARGS[@]}"; printf '\n' ;;
  start) lock_state; start_server ;;
  stop) lock_state; stop_server "$@" ;;
  restart) lock_state; require_config; stop_server; start_server ;;
  status) status_server ;;
  logs) [[ -f "$LOG_FILE" ]] || die '尚无日志'; if [[ "${1:-}" == -f ]]; then tail -n 60 -F "$LOG_FILE"; else tail -n 60 "$LOG_FILE"; fi ;;
  test) test_api ;;
  all) require_config; lock_state; install_runtime; convert_model; start_server ;;
  *) usage; die "未知命令: $cmd" ;;
esac
