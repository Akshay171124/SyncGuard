#!/bin/bash
# Gate G1: verify restored dataset counts and write the dataset manifest.
# Counts are recorded, not asserted — April's numbers are reference points, and
# the spec makes exact parity a non-goal. An order-of-magnitude miss means an
# extraction failed and must be investigated before preprocessing burns hours.
set -uo pipefail
cd /scratch/$USER/SyncGuard || exit 1

echo "=== G1: raw dataset counts ==="
printf "%-14s %10s  %s\n" DATASET FILES "REFERENCE (April)"
for d in fakeavceleb celebdf lrs2 avspeech; do
    n=$(find "data/raw/$d" -type f \( -name '*.mp4' -o -name '*.avi' \) 2>/dev/null | wc -l)
    case $d in
      fakeavceleb) ref="21,544 clips" ;;
      celebdf)     ref="(several thousand)" ;;
      lrs2)        ref="~96,000 clips" ;;
      avspeech)    ref="24,760 clips" ;;
    esac
    printf "%-14s %10s  %s\n" "$d" "$n" "$ref"
done

echo
echo "=== all files (any extension) per dataset ==="
for d in fakeavceleb celebdf lrs2 avspeech; do
    printf "%-14s %10s\n" "$d" "$(find "data/raw/$d" -type f 2>/dev/null | wc -l)"
done

echo
echo "=== disk usage ==="
du -sh data/raw/*/ 2>/dev/null

echo
echo "=== writing data/MANIFEST.json ==="
$HOME/.conda/envs/syncguard/bin/python - <<'PY'
import hashlib, json
from pathlib import Path
VID = {".mp4", ".avi"}
m = {}
for d in ["fakeavceleb", "celebdf", "lrs2", "avspeech", "dfdc"]:
    root = Path(f"data/raw/{d}")
    if not root.exists():
        m[d] = {"count": 0, "sha256": None, "note": "not present"}
        continue
    names = sorted(p.name for p in root.rglob("*") if p.suffix.lower() in VID)
    m[d] = {"count": len(names),
            "sha256": hashlib.sha256("".join(names).encode()).hexdigest()}
Path("data").mkdir(exist_ok=True)
Path("data/MANIFEST.json").write_text(json.dumps(m, indent=2))
print(json.dumps(m, indent=2))
PY
