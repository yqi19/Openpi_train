#!/usr/bin/env bash
set -euo pipefail

# Wait until:
# 1) tmux session "gpu" 的 openpi 流程切换到 occupy 阶段（训练结束），并且
# 2) /root/workspace/data_gen/verb_object_500_0325/ours.zip 文件传输完成（文件大小在一段时间内不再变化）
# 然后运行 run_overall_7.sh

TMUX_SESSION="${TMUX_SESSION:-gpu}"
OPENPI_DIR="${OPENPI_DIR:-/root/workspace/openpi}"

ZIP_PATH="${ZIP_PATH:-/root/workspace/data_gen/verb_object_500_0325/ours.zip}"
RUN_SCRIPT="${RUN_SCRIPT:-run_overall_7.sh}"

# openpi 训练进程结束判据（注意：occupy_gpu.py 会常驻占显存，不能用显存判断）
TRAIN_PROC_REGEX="${TRAIN_PROC_REGEX:-scripts/train_pytorch.py}"
OCCUPY_PROC_REGEX="${OCCUPY_PROC_REGEX:-occupy_gpu.py}"

POLL_INTERVAL_SECS="${POLL_INTERVAL_SECS:-5}"

# openpi 训练结束后是否立即杀掉 occupy_gpu.py
KILL_OCCUPY_ON_DONE="${KILL_OCCUPY_ON_DONE:-1}" # 1=kill, 0=不kill

# 等待 occupy_gpu.py 退出的超时时间（秒）
KILL_OCCUPY_TIMEOUT_SECS="${KILL_OCCUPY_TIMEOUT_SECS:-60}"

# 文件“传输结束”判据：文件大小稳定持续稳定多久
FILE_STABLE_INTERVAL_SECS="${FILE_STABLE_INTERVAL_SECS:-2}"
FILE_STABLE_SECONDS="${FILE_STABLE_SECONDS:-60}"
FILE_MIN_SIZE_BYTES="${FILE_MIN_SIZE_BYTES:-1}"

TIMEOUT_SECS="${TIMEOUT_SECS:-0}" # 0 表示不超时

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

is_cmd_running() {
  # pgrep -f 匹配整行 cmd；这里用 regex 过滤到目标 python 进程
  pgrep -f "$1" >/dev/null 2>&1
}

kill_cmd_running() {
  # kill 所有匹配的进程（用于简单场景：只要不是常驻进程即可）
  # shellcheck disable=SC2001
  local regex="$1"
  local pids
  pids="$(pgrep -f "$regex" || true)"
  if [[ -z "$pids" ]]; then
    return 0
  fi

  # 先用 TERM 给进程退出机会；如果不退出再用 KILL
  kill $pids >/dev/null 2>&1 || true

  local start_ts
  start_ts="$(date +%s)"
  while is_cmd_running "$regex"; do
    sleep 1
    if [[ "$KILL_OCCUPY_TIMEOUT_SECS" -gt 0 ]]; then
      local now elapsed
      now="$(date +%s)"
      elapsed=$((now - start_ts))
      if [[ "$elapsed" -ge "$KILL_OCCUPY_TIMEOUT_SECS" ]]; then
        log "occupy 退出超时，尝试强制 kill（regex='$regex'）"
        local force_pids
        force_pids="$(pgrep -f "$regex" || true)"
        if [[ -n "$force_pids" ]]; then
          kill -9 $force_pids >/dev/null 2>&1 || true
        fi
        break
      fi
    fi
  done
}

wait_tmux_ready() {
  while true; do
    if tmux has-session -t "$TMUX_SESSION" >/dev/null 2>&1; then
      return 0
    fi
    log "tmux session '$TMUX_SESSION' 不存在，等待中..."
    sleep "$POLL_INTERVAL_SECS"
  done
}

wait_openpi_done() {
  # openpi done：训练脚本不再运行 + occupy_gpu.py 仍在运行
  local start_ts
  start_ts="$(date +%s)"

  while true; do
    local train_running=0
    local occupy_running=0

    if is_cmd_running "$TRAIN_PROC_REGEX"; then
      train_running=1
    fi
    if is_cmd_running "$OCCUPY_PROC_REGEX"; then
      occupy_running=1
    fi

    if [[ "$train_running" -eq 0 && "$occupy_running" -eq 1 ]]; then
      log "openpi 训练已结束（$TRAIN_PROC_REGEX 不在运行），且占用进程已启动（$OCCUPY_PROC_REGEX 在运行）。"

      if [[ "$KILL_OCCUPY_ON_DONE" -eq 1 ]]; then
        log "按需杀掉 occupy 进程：'$OCCUPY_PROC_REGEX'"
        kill_cmd_running "$OCCUPY_PROC_REGEX"
        log "occupy 进程已停止。"
      else
        log "KILL_OCCUPY_ON_DONE=0，未杀掉 occupy 进程。"
      fi

      return 0
    fi

    if [[ "$TIMEOUT_SECS" -gt 0 ]]; then
      local now elapsed
      now="$(date +%s)"
      elapsed=$((now - start_ts))
      if [[ "$elapsed" -ge "$TIMEOUT_SECS" ]]; then
        log "超时：等待 openpi 结束超过 ${TIMEOUT_SECS}s。"
        exit 1
      fi
    fi

    log "等待 openpi 结束：train_running=$train_running, occupy_running=$occupy_running（每 ${POLL_INTERVAL_SECS}s 检查一次）"
    sleep "$POLL_INTERVAL_SECS"
  done
}

wait_zip_transfer_done() {
  local start_ts
  start_ts="$(date +%s)"

  local last_size=""
  local last_change_ts=""

  while true; do
    local now size
    now="$(date +%s)"

    if [[ -f "$ZIP_PATH" ]]; then
      size="$(stat -c '%s' "$ZIP_PATH" 2>/dev/null || echo 0)"
    else
      size=0
    fi

    if [[ "$size" -ge "$FILE_MIN_SIZE_BYTES" ]]; then
      if [[ "$size" != "$last_size" ]]; then
        last_size="$size"
        last_change_ts="$now"
      fi

      if [[ -n "${last_change_ts:-}" ]]; then
        local stable_for=$((now - last_change_ts))
        if [[ "$stable_for" -ge "$FILE_STABLE_SECONDS" ]]; then
          log "检测到 '$ZIP_PATH' 文件大小已稳定 ${stable_for}s（size=$size）。"
          return 0
        fi
      fi
    fi

    if [[ "$TIMEOUT_SECS" -gt 0 ]]; then
      local elapsed=$((now - start_ts))
      if [[ "$elapsed" -ge "$TIMEOUT_SECS" ]]; then
        log "超时：等待 '$ZIP_PATH' 传输完成超过 ${TIMEOUT_SECS}s。"
        exit 1
      fi
    fi

    sleep "$FILE_STABLE_INTERVAL_SECS"
  done
}

main() {
  log "开始等待：tmux='$TMUX_SESSION' openpi 结束 + zip 传输完成，然后运行 '${OPENPI_DIR}/${RUN_SCRIPT}'"

  # 可选：先等 tmux session 存在，避免误判环境还没启动
  wait_tmux_ready

  # openpi done 可能会比较久，这里直接阻塞等待直到满足条件
  wait_openpi_done

  wait_zip_transfer_done

  log "触发执行：${OPENPI_DIR}/${RUN_SCRIPT}"
  cd "$OPENPI_DIR"
  bash "$RUN_SCRIPT"
  log "脚本结束：run_overall_7.sh 已执行完成。"
}

main "$@"

