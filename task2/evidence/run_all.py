"""Re-run every defect-evidence script and rewrite its printout (<script>_result.txt; d4_order_test -> d4_order_result).

The scripts import the pipeline with sys.path '..', so each runs with this folder as its working directory."""
import subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for s in sorted(HERE.glob("d*.py")):
    out = HERE / (s.stem.removesuffix("_test") + "_result.txt")
    print(f"{s.name} -> {out.name}", flush=True)
    with open(out, "w") as f:
        r = subprocess.run([sys.executable, "-W", "ignore", s.name], cwd=HERE, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        sys.exit(f"{s.name} failed, see {out.name}")
