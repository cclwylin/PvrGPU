#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
generated_paths=(
    ".venv"
    "build"
    "out"
    "tmp"
    "tools/__pycache__"
    "tests/__pycache__"
)

failed=0
for relative_path in "${generated_paths[@]}"; do
    if [[ -e "${project_dir}/${relative_path}" ]]; then
        echo "Generated path found in source tree: ${relative_path}" >&2
        failed=1
    fi
done
if find "${project_dir}" -name '*.pyc' -print -quit | grep -q .; then
    echo "Python bytecode found in source tree." >&2
    failed=1
fi
if find "${project_dir}" -name '.DS_Store' -print -quit | grep -q .; then
    echo "Finder metadata found in source tree." >&2
    failed=1
fi
while IFS= read -r -d '' executable; do
    file_kind="$(file -b "${executable}")"
    case "${file_kind}" in
        *Mach-O*|*ELF*|*PE32*)
            echo "Compiled executable found in source tree: ${executable#"${project_dir}/"}" >&2
            failed=1
            ;;
    esac
done < <(find "${project_dir}" -type f -perm -111 -print0)
if find "${project_dir}" \
    \( -type f \( -name '*.o' -o -name '*.a' -o -name '*.so' -o -name '*.dylib' \) \
       -o -type d -name '*.dSYM' \) -print -quit | grep -q .; then
    echo "Compiled object/library/debug bundle found in source tree." >&2
    failed=1
fi
if ((failed)); then
    exit 1
fi

echo "Source tree guard PASS: generated files are outside iCloud."
