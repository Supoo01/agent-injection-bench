#!/usr/bin/bash
# Staendiger Injektions-Pruefstand (14.10.2026): eine neue Version gegen die Produktions-Referenz pruefen.
#
#   kandidat.sh start NAME MODELL [--referenz REF] [--leser] [--warte-stunden 24]
#       legt die Versionen an, startet sich als Unit injection-bench-candidate-NAME und kehrt sofort zurueck.
#   kandidat.sh plan  NAME MODELL [...]   nur anzeigen: Versionen, fehlende Laeufe, Wache (ohne GPU)
#   kandidat.sh lauf  NAME MODELL [...]   (in der Unit) warten bis GPU frei -> fehlende Laeufe messen -> Urteil
#
# MODELL: adapter path relative to PRUEFSTAND_HARNESS (as in versionen/*.json) or "basis".
# The reference (--referenz, default "referenz") needs versionen/<REF>_aus.json (create it with kandidat.py anlegen). Missing reference runs
# mit demselben Plan werden mitgemessen (einmalig, danach wiederverwendet).
# Gemessen: NAME_aus (Gateway aus, Modell-Robustheit) + NAME_s2s5_mensch0 (Sperre wie Testinstanz).
# Verdict/rules: pruefstand/kandidat.py; report pruefstand/berichte/, status line $PRUEFSTAND_STATUS_DIR/injektion_kandidat_letzter.json,
# optional notification via PRUEFSTAND_NOTIFY_CMD. Never two GPU jobs at once.
set -u
P="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$P")"
PY="$ROOT/venv/bin/python"
HARNESS=${PRUEFSTAND_HARNESS:-$ROOT/adapters}
PUNIT=${PRUEFSTAND_UNIT:-injection-bench}

cmd=${1:-}; shift || true
NAME=${1:-}; MODELL=${2:-}; shift 2 || { sed -n 2,15p "$0"; exit 1; }
REF=referenz; LESER=""; WARTE_H=24
while [ $# -gt 0 ]; do
    case $1 in
        --referenz) REF=$2; shift 2 ;;
        --leser) LESER=--leser; shift ;;
        --warte-stunden) WARTE_H=$2; shift 2 ;;
        *) echo "unbekannt: $1" >&2; exit 1 ;;
    esac
done
case $NAME in ''|*[!a-z0-9_]*) echo "NAME nur a-z0-9_" >&2; exit 1 ;; esac
UNIT=injection-bench-candidate-$NAME

notify() { [ -n "${PRUEFSTAND_NOTIFY_CMD:-}" ] && { $PRUEFSTAND_NOTIFY_CMD "[injektion] $*" >/dev/null 2>&1 || true; }; return 0; }
log() { echo "$(date '+%F %T') $*"; }

ref_modell() {
    [ -f "$P/versionen/${REF}_aus.json" ] || { echo "Referenz $REF fehlt: versionen/${REF}_aus.json anlegen" >&2; return 1; }
    "$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['modell'])" "$P/versionen/${REF}_aus.json"
}

vorbereiten() {
    cd "$ROOT" || exit 1
    [ "$MODELL" = basis ] || [ -f "$HARNESS/$MODELL/adapter_config.json" ] || { echo "kein Adapter $HARNESS/$MODELL" >&2; exit 1; }
    "$PY" -m pruefstand.kandidat anlegen "$NAME" "$MODELL" $LESER >/dev/null || exit 1
    ref_modell >/dev/null || exit 1
    PLAN=$("$PY" -m pruefstand.plan schnell | "$PY" -c "import json,sys; print(json.load(sys.stdin)['plan_sha256'])") || exit 1
    FEHLT=$("$PY" -m pruefstand.kandidat fehlend "$NAME" --referenz "$REF" --plan "$PLAN")
}

case $cmd in
plan)
    vorbereiten
    echo "Kandidat $NAME ($MODELL${LESER:+, Leser V1}) gegen $REF ($(ref_modell)), Plan ${PLAN:0:12}"
    echo "zu messen: ${FEHLT:-nichts (nur Urteil)}"
    [ -n "$FEHLT" ] && "$P/run.sh" plan --profil schnell $FEHLT | tail -n +2
    exit 0 ;;
start)
    vorbereiten
    systemctl --user is-active --quiet "$UNIT.service" && { echo "$UNIT laeuft schon"; exit 2; }
    exec systemd-run --user --unit "$UNIT" --collect -p MemoryMax=2G -p RuntimeMaxSec=3d \
        "$P/kandidat.sh" lauf "$NAME" "$MODELL" --referenz "$REF" $LESER --warte-stunden "$WARTE_H" ;;
lauf) ;;
*) sed -n 2,15p "$0"; exit 1 ;;
esac

# ------------------------------------------------------------------ lauf (in der Unit)
vorbereiten
log "Kandidat $NAME gegen $REF, Plan ${PLAN:0:12}, zu messen: ${FEHLT:-nichts}"
if [ -n "$FEHLT" ]; then
    t_ende=$(( $(date +%s) + WARTE_H * 3600 ))
    until g=$("$P/wache.sh") && ! systemctl --user is-active --quiet "$PUNIT.service"; do
        [ "$(date +%s)" -lt "$t_ende" ] || { log "GPU nach $WARTE_H h nicht frei: $(echo $g)"; notify "$NAME: GPU nach $WARTE_H h nicht frei – nicht gemessen"; exit 3; }
        sleep 300
    done
    log "GPU frei, starte Pruefstand: $FEHLT"
    "$P/run.sh" start --profil schnell $FEHLT || { log "run.sh start fehlgeschlagen"; notify "$NAME: Pruefstand-Start fehlgeschlagen"; exit 4; }
    sleep 30
    while systemctl --user is-active --quiet "$PUNIT.service"; do sleep 60; done
    log "Pruefstand beendet"
fi
cd "$ROOT" || exit 1
out=$("$PY" -m pruefstand.kandidat pruefen "$NAME" --referenz "$REF" 2>&1); rc=$?
log "$out"
notify "$(echo "$out" | head -1)"
exit $rc
