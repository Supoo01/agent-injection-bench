#!/usr/bin/bash
# Injektions-Pruefstand (AgentDojo 0.1.35 + Hermes-Gateway-Bruecke): jede Version gegen Prompt-Injektion testen.
#
#   run.sh plan  [--profil schnell|voll] VERSION...   Plan, Schaetzung, Versionserfassung (ohne GPU), Wache-Status
#   run.sh start [--profil schnell|voll] VERSION...   Wache, dann als systemd-run --user Unit $PRUEFSTAND_UNIT (default injection-bench)
#   run.sh lauf  [--profil schnell|voll] VERSION...   (in der Unit) je Version: erfassen -> Container -> Messung -> Auswertung
#
# VERSION = Datei pruefstand/versionen/<name>.json oder Pfad zu einer Spezifikation (siehe pruefstand/version.py).
# Ablage: pruefstand/laeufe/<Zeit>_<profil>/<name>_<version_id>/ ; Verlauf pruefstand/verlauf.jsonl + uebersicht.md;
# Status line: $PRUEFSTAND_STATUS_DIR/pruefstand_letzter.json (default pruefstand/status/).
# Start guard: pruefstand/wache.sh (own container prefix, optional busy units / production unit, port 8012).
# Notifications: optional command PRUEFSTAND_NOTIFY_CMD (gets the message as argument). Network: localhost:8012 only.
# Nie: Produktion anfassen, fremde Container stoppen, pushen.
set -u
P="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$P")"
PY="$ROOT/venv/bin/python"
UNIT=${PRUEFSTAND_UNIT:-injection-bench}
VUNIT=${PRUEFSTAND_VLLM_UNIT:-injection-bench-vllm}
CONTAINER=${PRUEFSTAND_CONTAINER:-injection-bench-vllm}
WORKERS=${PRUEFSTAND_WORKERS:-8}
export PRUEFSTAND_EIGENE="${PRUEFSTAND_EIGENE:-} $UNIT"

cmd=${1:-}; shift || true
PROFIL=schnell
VERS=()
while [ $# -gt 0 ]; do
    case $1 in
        --profil) PROFIL=$2; shift 2 ;;
        -h|--help) sed -n 2,14p "$0"; exit 0 ;;
        *) VERS+=("$1"); shift ;;
    esac
done
case $PROFIL in schnell|voll) ;; *) echo "Profil $PROFIL unbekannt" >&2; exit 1 ;; esac
[ ${#VERS[@]} -gt 0 ] || { echo "keine Version angegeben" >&2; exit 1; }

spec_pfad() {
    case $1 in
        */*|*.json) [ -f "$1" ] && { readlink -f "$1"; return 0; } ;;
        *) [ -f "$P/versionen/$1.json" ] && { echo "$P/versionen/$1.json"; return 0; } ;;
    esac
    echo "Versionsdatei $1 nicht gefunden" >&2; return 1
}
SPECS=()
for v in "${VERS[@]}"; do s=$(spec_pfad "$v") || exit 1; SPECS+=("$s"); done

notify() { [ -n "${PRUEFSTAND_NOTIFY_CMD:-}" ] && { $PRUEFSTAND_NOTIFY_CMD "[pruefstand] $*" >/dev/null 2>&1 || true; }; return 0; }

case $cmd in
plan)
    cd "$ROOT" || exit 1
    "$PY" -m pruefstand.plan "$PROFIL" --versionen ${#SPECS[@]}
    for s in "${SPECS[@]}"; do
        echo "== $(basename "$s")"
        "$PY" - "$s" <<'EOF' || echo "   VERSIONSFEHLER"
import json, sys
from pruefstand.version import erfasse, kurzbeschreibung
r = erfasse(json.load(open(sys.argv[1])))
print("  ", r["version_id"], kurzbeschreibung(r))
EOF
    done
    echo "== Wache:"; "$P/wache.sh" && echo "   frei"
    exit 0 ;;
start)
    if ! g=$("$P/wache.sh"); then echo "wartet auf GPU: $(echo $g)"; exit 2; fi
    systemctl --user is-active --quiet "$UNIT.service" && { echo "$UNIT laeuft schon"; exit 2; }
    exec systemd-run --user --unit "$UNIT" --collect -p MemoryMax=16G -p RuntimeMaxSec=4d \
        --setenv=PRUEFSTAND_EIGENE="$PRUEFSTAND_EIGENE" "$P/run.sh" lauf --profil "$PROFIL" "${SPECS[@]}" ;;
lauf) ;;
*) echo "Aufruf: run.sh plan|start|lauf [--profil schnell|voll] VERSION..." >&2; exit 1 ;;
esac

# ------------------------------------------------------------------ lauf
cd "$ROOT" || exit 1
STAMP=$(date +%Y%m%d_%H%M)
BASIS="$P/laeufe/${STAMP}_$PROFIL"
mkdir -p "$BASIS"
LOG="$BASIS/ablauf.log"
exec > >(tee -a "$LOG") 2>&1
log() { echo "$(date '+%F %T') $*"; }

started=0
stop_container() {
    [ "$started" = 1 ] || return 0
    log "stoppe $VUNIT / $CONTAINER"
    systemctl --user stop "$VUNIT.service" 2>/dev/null || true
    sleep 5
    if timeout 60 docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then timeout 120 docker stop "$CONTAINER" || true; fi
    for _ in $(seq 1 36); do
        timeout 60 docker ps --format '{{.Names}}' | grep -qx "$CONTAINER" || { started=0; return 0; }
        sleep 5
    done
    log "WARNUNG: $CONTAINER laeuft nach 3 min noch"; return 1
}
trap 'stop_container' EXIT
trap 'log "Signal -- Ende"; notify "Abbruch durch Signal ($BASIS)"; exit 143' TERM INT

if ! g=$("$P/wache.sh"); then log "Start-Wache: belegt: $(echo $g)"; notify "Nicht gestartet, GPU belegt: $(echo $g)"; exit 2; fi
"$PY" -m pruefstand.plan "$PROFIL" --out "$BASIS/plan.json" --versionen ${#SPECS[@]} > "$BASIS/plan_info.json" || { log "Plan fehlgeschlagen"; exit 3; }
FRIST=$("$PY" -c "from pruefstand.plan import PROFILES; print(PROFILES['$PROFIL']['frist_min'])")
log "Pruefstand $PROFIL: ${#SPECS[@]} Version(en), Plan $(grep -o '"plan_sha256": "[0-9a-f]*' "$BASIS/plan_info.json" | cut -c17-28), Frist je Version $FRIST min"
notify "Start Prüfstand ($PROFIL, ${#SPECS[@]} Versionen: $(for s in "${SPECS[@]}"; do basename "$s" .json; done | xargs echo)). Ablage $BASIS"

t_all=$(date +%s)
ok=0; fehl=0
zeilen=()
for s in "${SPECS[@]}"; do
    name=$(basename "$s" .json)
    tmpv="$BASIS/.version_$name.json"
    if ! "$PY" -m pruefstand.version "$s" --out "$tmpv" >/dev/null; then
        log "$name: Versionserfassung fehlgeschlagen -- uebersprungen"; notify "$name: Versionserfassung fehlgeschlagen"; fehl=$((fehl+1)); continue
    fi
    vid=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['version_id'])" "$tmpv")
    modell=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['spec']['modell'])" "$tmpv")
    D="$BASIS/${name}_$vid"; mkdir -p "$D"; mv "$tmpv" "$D/version.json"; cp "$BASIS/plan.json" "$D/plan.json"
    if ! g=$("$P/wache.sh"); then log "$name: GPU belegt vor Containerstart: $(echo $g) -- Abbruch"; notify "Abbruch vor $name: GPU belegt ($(echo $g))"; fehl=$((fehl+1)); break; fi
    vorlage=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['spec'].get('chat_template') or '')" "$D/version.json")
    log "$name ($vid): Container mit $modell${vorlage:+, Chat-Vorlage $vorlage}"
    PRUEFSTAND_CHAT_TEMPLATE="$vorlage" "$P/start_vllm.sh" "$modell" || { log "$name: Containerstart fehlgeschlagen"; notify "$name: Containerstart fehlgeschlagen"; fehl=$((fehl+1)); continue; }
    started=1
    want=coding; [ "$modell" != basis ] && want=hermes
    bereit=0
    for _ in $(seq 1 180); do
        if curl -sf --max-time 5 http://localhost:8012/v1/models 2>/dev/null | grep -q "\"$want\""; then bereit=1; break; fi
        systemctl --user is-active --quiet "$VUNIT.service" || break
        sleep 10
    done
    if [ $bereit != 1 ]; then log "$name: vLLM nicht bereit"; notify "$name: vLLM nicht bereit"; stop_container; fehl=$((fehl+1)); continue; fi
    log "$name: vLLM bereit, Messung ($WORKERS parallel)"
    t0=$(date +%s)
    timeout $(( FRIST * 60 + 900 )) "$PY" -m pruefstand.lauf --version "$D/version.json" --plan "$D/plan.json" --out "$D" \
        --workers "$WORKERS" --frist-min "$FRIST" > "$D/lauf.log" 2>&1
    rc=$?
    stop_container
    log "$name: Messung rc=$rc nach $(( ($(date +%s) - t0) / 60 )) min"
    if [ -f "$D/lauf_meta.json" ] && z=$("$PY" -m pruefstand.auswertung "$D"); then
        log "$name: $z"; zeilen+=("$z"); notify "$z"; ok=$((ok+1))
    else
        log "$name: Auswertung fehlgeschlagen (rc $rc), siehe $D/lauf.log"; notify "$name: Messung/Auswertung fehlgeschlagen (rc $rc)"; fehl=$((fehl+1))
    fi
done
mins=$(( ($(date +%s) - t_all) / 60 ))
log "Ende nach $mins min: $ok ok, $fehl fehlgeschlagen"
notify "Prüfstand fertig nach $mins min: $ok ok, $fehl fehlgeschlagen. Übersicht $P/uebersicht.md"
[ $fehl -eq 0 ]
