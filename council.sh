#!/usr/bin/env bash
# Démarre / arrête PromptCouncil en arrière-plan.
# Usage : ./council.sh {start|stop|restart|status}
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$DIR/.council.pid"
LOG_FILE="$DIR/council.log"
PORT="${PORT:-8501}"
ADDRESS="${ADDRESS:-localhost}"   # ADDRESS=0.0.0.0 pour exposer sur le réseau

is_running() {
  [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

start() {
  if is_running; then
    echo "PromptCouncil tourne déjà (PID $(cat "$PID_FILE")) → http://localhost:$PORT"
    return
  fi
  if [[ ! -x "$DIR/.venv/bin/streamlit" ]]; then
    echo "Environnement manquant. Lancez d'abord :"
    echo "  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
  fi
  cd "$DIR"
  nohup .venv/bin/streamlit run app.py --server.headless true \
    --server.port "$PORT" --server.address "$ADDRESS" >"$LOG_FILE" 2>&1 &
  echo $! >"$PID_FILE"

  for _ in $(seq 30); do
    if curl -s "http://localhost:$PORT/_stcore/health" >/dev/null 2>&1; then
      echo "✅ PromptCouncil démarré (PID $(cat "$PID_FILE")) → http://localhost:$PORT"
      return
    fi
    if ! is_running; then break; fi
    sleep 0.5
  done
  echo "❌ Échec du démarrage, voir $LOG_FILE :"
  tail -n 20 "$LOG_FILE"
  rm -f "$PID_FILE"
  exit 1
}

stop() {
  if ! is_running; then
    echo "PromptCouncil n'est pas lancé."
    rm -f "$PID_FILE"
    return
  fi
  local pid
  pid="$(cat "$PID_FILE")"
  kill "$pid"
  for _ in $(seq 20); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.25
  done
  kill -0 "$pid" 2>/dev/null && kill -9 "$pid"
  rm -f "$PID_FILE"
  echo "🛑 PromptCouncil arrêté."
}

status() {
  if is_running; then
    echo "🟢 En cours (PID $(cat "$PID_FILE")) → http://localhost:$PORT"
  else
    echo "⚪ Arrêté."
  fi
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
  status) status ;;
  *) echo "Usage : $0 {start|stop|restart|status}"; exit 1 ;;
esac
