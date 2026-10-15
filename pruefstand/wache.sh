#!/usr/bin/bash
# GPU guard: never run two GPU jobs at once. Output: one line per reason; exit 0 = free, 2 = busy.
# Busy while
#   - a container whose name starts with PRUEFSTAND_CONTAINER_PREFIX runs (default: injection-bench-vllm),
#   - an active systemd --user service matches PRUEFSTAND_BUSY_UNITS (extended regex, default: none),
#     except the units listed in PRUEFSTAND_EIGENE (own run),
#   - PRUEFSTAND_PROD_UNIT (optional) is active (e.g. a production vLLM service on the same GPU),
#   - port PRUEFSTAND_PORT (default 8012) is in use,
#   - PRUEFSTAND_REQUIRE_FILE (optional) is missing.
# Any error while checking counts as busy (safe side).
set -u
EIGENE=" ${PRUEFSTAND_EIGENE:-} "
PREFIX=${PRUEFSTAND_CONTAINER_PREFIX:-injection-bench-vllm}
MUSTER=${PRUEFSTAND_BUSY_UNITS:-}
PORT=${PRUEFSTAND_PORT:-8012}
grund=()

if c=$(timeout 60 docker ps --format '{{.Names}}' 2>/dev/null); then
    c=$(printf '%s\n' "$c" | grep -E "^$PREFIX" | xargs echo)
    [ -n "$c" ] && grund+=("container=$c")
else
    grund+=("docker-ps-fehlgeschlagen")
fi

if [ -n "$MUSTER" ]; then
    if u=$(timeout 30 systemctl --user list-units --type=service --state=active,activating,reloading --no-legend --plain 2>/dev/null); then
        for name in $(printf '%s\n' "$u" | awk '{print $1}' | grep -E "$MUSTER" || true); do
            base=${name%.service}
            case "$EIGENE" in *" $base "*|*" $name "*) continue ;; esac
            grund+=("unit=$name")
        done
    else
        grund+=("systemctl-fehlgeschlagen")
    fi
fi

if [ -n "${PRUEFSTAND_PROD_UNIT:-}" ]; then
    s=$(timeout 30 systemctl --user is-active "$PRUEFSTAND_PROD_UNIT" 2>/dev/null); s=${s:-unbekannt}
    case $s in inactive|failed) ;; *) grund+=("produktion=$PRUEFSTAND_PROD_UNIT:$s") ;; esac
fi

if p=$(ss -ltn 2>/dev/null); then
    printf '%s\n' "$p" | grep -qE "[:.]$PORT[[:space:]]" && grund+=("port$PORT=belegt")
else
    grund+=("ss-fehlgeschlagen")
fi

if [ -n "${PRUEFSTAND_REQUIRE_FILE:-}" ] && [ ! -f "$PRUEFSTAND_REQUIRE_FILE" ]; then
    grund+=("datei_fehlt=$PRUEFSTAND_REQUIRE_FILE")
fi

if [ ${#grund[@]} -gt 0 ]; then
    printf '%s\n' "${grund[@]}"
    exit 2
fi
exit 0
