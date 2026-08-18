#!/usr/bin/env bash
# 启动全部数据集下载(每个数据集一个独立 tmux 会话,断网/掉线不受影响)。
#
# 用法:
#   bash tools/data/download_all.sh            # 先检查存储,再逐个启动 tmux 下载会话
#   bash tools/data/download_all.sh --check    # 只检查存储,不启动下载
#
# 会话名: dl-cocotext / dl-hiertext / dl-textocr
# 日志:   /home/heqi/dataset/logs/<会话名>.log (tee 追加)
set -euo pipefail

BASE_DIR="${DATASET_ROOT:-/home/heqi/dataset}"
LOG_DIR="$BASE_DIR/logs"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 需求估算(实测 content-length + 解压后空间):
#   cocotext: 13.5GB(train2014.zip) + 1.3GB(annot) + ~20GB(解压) ≈ 35GB
#   hiertext: 3.9GB(tgz) + ~12GB(解压) ≈ 16GB
#   textocr:  7.1GB(zip) + ~30GB(解压) ≈ 37GB
# 合计 ≈ 88GB,按 90GB 保守检查。
REQUIRED_MB=$(( 90 * 1024 ))

mkdir -p "$BASE_DIR" "$LOG_DIR"

avail_kb=$(df -Pk "$BASE_DIR" | awk 'NR==2 {print $4}')
avail_mb=$(( avail_kb / 1024 ))
echo "存储检查: 目标目录 $BASE_DIR"
echo "  可用:   $(( avail_mb / 1024 )) GiB"
echo "  需求:   $(( REQUIRED_MB / 1024 )) GiB (下载 26GB + 解压 ~62GB 的保守估计)"
if [ "$avail_mb" -lt "$REQUIRED_MB" ]; then
    echo "ERROR: 可用存储不足(${avail_mb}MB < ${REQUIRED_MB}MB),请先扩容或调整 DATASET_ROOT" >&2
    exit 1
fi
echo "  存储充足,继续。"

if [ "${1:-}" = "--check" ]; then
    echo "(--check 模式,不启动下载)"
    exit 0
fi

command -v tmux >/dev/null 2>&1 || { echo "ERROR: tmux 未安装" >&2; exit 1; }

for spec in "dl-cocotext:download_cocotext.sh" "dl-hiertext:download_hiertext.sh" "dl-textocr:download_textocr.sh"; do
    name="${spec%%:*}"
    script="${spec##*:}"
    if tmux has-session -t "$name" 2>/dev/null; then
        echo "tmux 会话 $name 已存在,跳过"
        continue
    fi
    tmux new-session -d -s "$name" "bash '$SCRIPT_DIR/$script' 2>&1 | tee -a '$LOG_DIR/$name.log'"
    echo "已启动 tmux 会话 $name (日志: $LOG_DIR/$name.log)"
done

echo
echo "当前 tmux 会话:"
tmux ls
echo
echo "查看进度:  tmux attach -t <会话名>      (退出: Ctrl-b d)"
echo "查看日志:  tail -f $LOG_DIR/<会话名>.log"
