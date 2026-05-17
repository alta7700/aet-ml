#!/usr/bin/env bash
# Проверка: пакет new_arch не должен импортировать ничего из внешних
# scripts/ и dataset_pipeline/. Локальный new_arch/dataset_pipeline разрешён.
#
# Запуск из корня репозитория:
#   bash new_arch/scripts/check_self_contained.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${ROOT}"

fail=0

# 1) Импорты вида `from scripts...` / `import scripts...` запрещены полностью.
if grep -rn --include="*.py" -E "^(from|import)[[:space:]]+scripts(\.|[[:space:]]|$)" "${TARGET}"; then
    echo "ERROR: new_arch импортирует из внешнего scripts/" >&2
    fail=1
fi

# 2) Импорты dataset_pipeline разрешены только когда это локальный пакет
#    внутри new_arch/dataset_pipeline. Внешний dataset_pipeline запрещён.
#    Так как импорты в new_arch написаны без префикса (`from dataset_pipeline...`),
#    отличить локальный от внешнего по тексту невозможно — но они подхватываются
#    через PYTHONPATH=new_arch, поэтому здесь достаточно убедиться, что локальный
#    пакет на месте.
if [[ ! -d "${TARGET}/dataset_pipeline" ]]; then
    if grep -rn --include="*.py" -E "^(from|import)[[:space:]]+dataset_pipeline(\.|[[:space:]]|$)" "${TARGET}"; then
        echo "ERROR: используется внешний dataset_pipeline (локальный new_arch/dataset_pipeline отсутствует)" >&2
        fail=1
    fi
fi

# 3) Любые `from new_arch...` / `import new_arch...` — признак старого стиля
#    (пакет должен быть самодостаточным и обращаться к себе по локальным именам).
if grep -rn --include="*.py" -E "^(from|import)[[:space:]]+new_arch(\.|[[:space:]]|$)" "${TARGET}"; then
    echo "ERROR: остались импорты new_arch.X — должны быть переименованы в локальные" >&2
    fail=1
fi

if (( fail == 0 )); then
    echo "OK: new_arch self-contained"
fi
exit "${fail}"
