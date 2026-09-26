"""Quick comparison table for one tag: cmp.py TAG [REF_ARM]."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from harness import judge
tag = sys.argv[1]; ref = sys.argv[2] if len(sys.argv) > 2 else None
R = {json.load(open(f))["arm"]: json.load(open(f)) for f in Path(__file__).parent.glob(f"results/*_{tag}_s*.json")}
b = R["baseline"]
for a, r in sorted(R.items(), key=lambda x: x[1]["mae"]):
    j = judge(b, r); s = f"{a:22s} mae {r['mae']:6.1f} p90 {r['p90_ae']:6.1f} bias {r['bias']:6.1f} dMAE {j['mae_change_pct']:+5.1f}% dp90 {j['p90_change_pct']:+5.1f}% wb {j['worst_bucket_change_pct']:+5.1f}% win {j['wins']}"
    if ref and ref in R and a != ref:
        jr = judge(R[ref], r); s += f" | vs {ref}: {jr['mae_change_pct']:+5.2f}% p90 {jr['p90_change_pct']:+5.2f}% wb {jr['worst_bucket_change_pct']:+5.2f}%"
    print(s)
