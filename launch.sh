#!/bin/bash
# launch.sh — Starts the pipeline in background with nohup.
# The process keeps running even after closing VSCode or the terminal.
#
# Usage:
#   bash launch.sh          → starts pipeline in background
#   bash launch.sh --follow → starts pipeline AND tails the log live

mkdir -p logs

LOGFILE="logs/pipeline_$(date +%Y%m%d_%H%M%S).log"
PIDFILE="logs/pipeline.pid"

echo ""
echo "  Starting pipeline in background …"
echo "  Log file : $LOGFILE"
echo ""

# nohup + python3 -u (unbuffered) ensures every line appears in the log
# immediately as it is printed, without buffering delays.
# setsid creates a new process group so we can kill the whole tree at once
nohup setsid bash run_all.sh > "$LOGFILE" 2>&1 &
PID=$!
echo $PID > "$PIDFILE"

echo "  PID      : $PID  (saved to $PIDFILE)"
echo ""
echo "  ┌─────────────────────────────────────────────────────┐"
echo "  │  Follow the log live:                               │"
echo "  │    tail -f $LOGFILE                                 │"
echo "  │                                                     │"
echo "  │  Check if still running:                            │"
echo "  │    ps -p $PID                                       │"
echo "  │                                                     │"
echo "  │  Stop the pipeline (kills ALL child processes):     │"
echo "  │    kill -- -$(cat logs/pipeline.pid 2>/dev/null)    │"
echo "  └─────────────────────────────────────────────────────┘"
echo ""

# If called with --follow, stay attached and tail the log
if [[ "$1" == "--follow" ]]; then
    echo "  Following log (Ctrl+C to detach — pipeline keeps running) …"
    echo ""
    tail -f "$LOGFILE"
fi
