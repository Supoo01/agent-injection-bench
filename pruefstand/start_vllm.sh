#!/usr/bin/bash
# Reference vLLM setup (NVIDIA Jetson Thor, Nemotron-3.5-Lightning NVFP4): test container for ONE version on port 8012.
# Adapt the docker/vLLM flags to your model and GPU.
#   start_vllm.sh basis                               -> Basismodell als 'coding'
#   start_vllm.sh data/lora_adapter_p4b_thinking_b    -> Basismodell 'coding' + Adapter als 'hermes' (ro eingehaengt)
# Same flags for ALL versions (only the adapter differs): --enable-lora --max-lora-rank 16 --max-loras 1
# --lora-modules hermes=..., max-model-len 65536, sampling from the generation config (PRUEFSTAND_GEN_CONFIG).
# No speculative decoding (not validated together with LoRA).
# Runs as systemd --user unit $PRUEFSTAND_VLLM_UNIT (MemoryMax=64G), container --rm with an exact name.
# No guard here: run.sh checks pruefstand/wache.sh first. Stop: systemctl --user stop $PRUEFSTAND_VLLM_UNIT
set -eu
MODELL=${1:?basis oder Adapter-Pfad (relativ zu PRUEFSTAND_HARNESS)}
NAME=${PRUEFSTAND_CONTAINER:-injection-bench-vllm}
UNIT=${PRUEFSTAND_VLLM_UNIT:-injection-bench-vllm}
IMAGE=${PRUEFSTAND_IMAGE:-vllm/vllm-openai:v0.27.1}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HARNESS=${PRUEFSTAND_HARNESS:-$ROOT/adapters}
GEN=${PRUEFSTAND_GEN_CONFIG:-$ROOT/config/generation_config.json}
MODELS=${PRUEFSTAND_MODELS:-$HOME/.cache/huggingface}
BASE_MODEL=${PRUEFSTAND_BASE_MODEL:-nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4}
case "$NAME" in injection-bench-vllm*) ;; *) echo "Containername $NAME nicht erlaubt" >&2; exit 2 ;; esac
LORA_MOUNT=()
LORA_MOD=()
if [ "$MODELL" != basis ]; then
    MODELL=${MODELL#"$HARNESS"/}
    case "$MODELL" in /*|*..*) echo "Adapter $MODELL muss relativ im Repo $HARNESS liegen" >&2; exit 1 ;; esac
    [ -f "$HARNESS/$MODELL/adapter_config.json" ] && [ -f "$HARNESS/$MODELL/adapter_model.safetensors" ] \
        || { echo "kein Adapter unter $HARNESS/$MODELL" >&2; exit 1; }
    LORA_MOUNT=(-v "$HARNESS/$MODELL":/adapters/hermes:ro)
    LORA_MOD=(--lora-modules hermes=/adapters/hermes)
fi
# E12 (07.10.): optionale Chat-Vorlage (PRUEFSTAND_CHAT_TEMPLATE, relativ zu PRUEFSTAND_HARNESS; setzt run.sh aus dem
# Versionsfeld chat_template). Gilt serverweit fuer 'coding' und 'hermes'. Leer = Vorlage des Basismodells (wie bisher).
VORLAGE=${PRUEFSTAND_CHAT_TEMPLATE:-}
VORLAGE_MOUNT=()
VORLAGE_ARG=()
if [ -n "$VORLAGE" ]; then
    VORLAGE=${VORLAGE#"$HARNESS"/}
    case "$VORLAGE" in /*|*..*) echo "Chat-Vorlage $VORLAGE muss relativ im Repo $HARNESS liegen" >&2; exit 1 ;; esac
    [ -f "$HARNESS/$VORLAGE" ] || { echo "Chat-Vorlage $HARNESS/$VORLAGE fehlt" >&2; exit 1; }
    VORLAGE_MOUNT=(-v "$HARNESS/$VORLAGE":/vorlagen/chat_template.jinja:ro)
    VORLAGE_ARG=(--chat-template /vorlagen/chat_template.jinja)
fi
[ -r "$GEN" ] || { echo "Generation-Config $GEN fehlt" >&2; exit 1; }
if [ "${PRUEFSTAND_DRY_RUN:-0}" = 1 ]; then RUN=(echo); else RUN=(); fi
exec "${RUN[@]}" systemd-run --user --unit "$UNIT" -p MemoryMax=64G --collect \
  docker run --rm --name "$NAME" --runtime=nvidia --network host --shm-size 8g \
  --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 \
  -v "$MODELS":/root/.cache/huggingface -v "$HOME"/.cache/vllm:/root/.cache/vllm \
  "${LORA_MOUNT[@]}" "${VORLAGE_MOUNT[@]}" \
  "$IMAGE" \
  --model "$BASE_MODEL" --served-model-name coding --port 8012 \
  --reasoning-parser nemotron_v3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --max-model-len 65536 --max-num-seqs 8 --kv-cache-dtype fp8 --gpu-memory-utilization 0.30 \
  --trust-remote-code --max-num-batched-tokens 16384 --enable-prefix-caching \
  --mamba-backend flashinfer --mamba-ssm-cache-dtype float16 --enable-mamba-cache-stochastic-rounding \
  --mamba-cache-philox-rounds 5 --mamba-cache-mode align \
  --enable-lora --max-lora-rank 16 --max-loras 1 "${LORA_MOD[@]}" "${VORLAGE_ARG[@]}" \
  --generation-config vllm --override-generation-config "$(cat "$GEN")"
