#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ARC-AGI Neural Fountain — specialize · LT · Master          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo

kind=""      # normal|spectrum
corpus=""    # 1|2|both
quick=""
all_fam=""
rest=()
for a in "$@"; do
  case "$a" in
    normal|mnist|master) kind="normal" ;;
    spectrum|layers|showcase) kind="spectrum" ;;
    # Digit alone: same as loom_neural_fountain (1=normal, 2=spectrum), default AGI-1
    1)
      if [[ -z "$kind" ]]; then
        kind="normal"
        corpus="${corpus:-1}"
      else
        corpus="1"
      fi
      ;;
    2)
      if [[ -z "$kind" ]]; then
        kind="spectrum"
        corpus="${corpus:-1}"
      else
        corpus="2"
      fi
      ;;
    arc1|agi1) corpus="1" ;;
    arc2|agi2) corpus="2" ;;
    both|all) corpus="both" ;;
    quick) quick="1" ;;
    transport|all-families) all_fam="1" ;;
    *) rest+=("$a") ;;
  esac
done

if [[ -z "$kind" ]]; then
  echo "What do you want to run?"
  echo
  echo "  1) normal    — dense Master like loom_neural_fountain ./run.sh 1"
  echo "                 (all train demos → LT → Master · EVAL held out)"
  echo "  2) spectrum  — family×dtype bake-off + mega fountain"
  echo
  read -r -p "Choose [1/2]: " choice
  case "${choice:-}" in
    1|normal|n|N) kind="normal" ;;
    2|spectrum|layers|l|L) kind="spectrum" ;;
    *)
      echo "invalid choice: ${choice:-<empty>} (expected 1 or 2)" >&2
      exit 2
      ;;
  esac
  echo
fi

if [[ -z "$corpus" ]]; then
  echo "Which ARC corpus?"
  echo "  1) ARC-AGI-1 (default)"
  echo "  2) ARC-AGI-2"
  echo "  3) both"
  read -r -p "Choose [1/2/3]: " c
  case "${c:-1}" in
    1) corpus="1" ;;
    2) corpus="2" ;;
    3|both) corpus="both" ;;
    *)
      echo "invalid corpus" >&2
      exit 2
      ;;
  esac
  echo
fi

if [[ "$kind" == "spectrum" && -z "$all_fam" && -t 0 && ${#rest[@]} -eq 0 ]]; then
  # Only prompt when interactive and no extra flags already chose families.
  if [[ -z "${FOUNTAIN_NO_PROMPT:-}" ]]; then
    echo "Spectrum families?"
    echo "  1) dense+residual (default) — ARC grid bake-off"
    echo "  2) all families             — + transport probes"
    read -r -p "Choose [1/2]: " fam
    case "${fam:-1}" in
      2|all|transport) all_fam="1" ;;
    esac
    echo
  fi
fi

if [[ -z "$quick" && -t 0 ]]; then
  if [[ "$kind" == "normal" ]]; then
    read -r -p "Quick? (fewer epochs · still ALL canvas-fit demos) [y/N]: " q
  else
    read -r -p "Quick? (7 dtypes · fewer epochs — full corpus) [y/N]: " q
  fi
  case "${q:-}" in y|Y|yes) quick="1" ;; esac
  echo
fi

args=("$kind")
[[ "$corpus" == "1" ]] && args+=(1)
[[ "$corpus" == "2" ]] && args+=(2)
[[ "$corpus" == "both" ]] && args+=(both)
[[ -n "$quick" ]] && args+=(quick)
[[ -n "$all_fam" ]] && args+=(-all-families)
args+=("${rest[@]}")

echo "→ go run . ${args[*]}"
if [[ "$kind" == "normal" ]]; then
  echo "  MNIST-style: dense specialists → LT → Master on all train demos (eval out)"
else
  echo "  spectrum bake-off + mega"
fi
echo
go run . "${args[@]}"

echo
echo "tips:"
echo "  ./run.sh 1                       # normal · AGI-1 (MNIST-style Master)"
echo "  ./run.sh 1 quick                 # K=8/3ep · still all train demos"
echo "  ./run.sh normal 2                # normal · AGI-2"
echo "  ./run.sh 2                       # spectrum · AGI-1"
echo "  ./run.sh spectrum 1 quick        # dense+residual × 7 dtypes"
echo "  logs/"
