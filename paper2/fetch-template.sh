#!/usr/bin/env bash
# Fetch Springer's LNCS class into this directory.
#
# `llncs.cls` is deliberately *not* committed. Springer distributes it for author
# use rather than under an open licence, and this repository's NOTICE asserts it
# carries no third-party code. Keeping the class out of git preserves both.
#
# Idempotent: exits early if the class is already present.
set -euo pipefail
cd "$(dirname "$0")"

if [[ -f llncs.cls ]]; then
    echo "llncs.cls already present; nothing to do."
    exit 0
fi

CTAN="https://mirrors.ctan.org/macros/latex/contrib/llncs.zip"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "Fetching LNCS class from CTAN..."
if ! curl -fsSL --retry 3 -m 120 -o "$tmp/llncs.zip" "$CTAN"; then
    cat >&2 <<'EOF'
ERROR: could not download the LNCS class.

Fetch it by hand instead -- Springer's "LaTeX Proceedings Templates" at
https://www.springer.com/gp/computer-science/lncs/conference-proceedings-guidelines
or CTAN at https://ctan.org/pkg/llncs -- and drop llncs.cls (and splncs04.bst,
if you want the Springer bibliography style) into this directory.
EOF
    exit 1
fi

# -j flattens, -o overwrites: the archive nests under llncs/.
unzip -joq "$tmp/llncs.zip" -d "$tmp/x"
for f in llncs.cls splncs04.bst; do
    if [[ -f "$tmp/x/$f" ]]; then
        cp "$tmp/x/$f" .
        echo "  installed $f"
    fi
done

if [[ ! -f llncs.cls ]]; then
    echo "ERROR: archive did not contain llncs.cls" >&2
    exit 1
fi
echo "Done. Now run: make"
