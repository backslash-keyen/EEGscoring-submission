"""Run every Task 2 ask script in order, without opening figure windows (main.py, part T2). Each prints its answer
and saves its figures to outputs/asks/. A failing script stops the run, so a broken answer cannot pass unnoticed.

python task2/asks/run_all.py            all of them
python task2/asks/run_all.py T2c        only the ones whose name starts with T2c
"""
import os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
prefix = sys.argv[1] if len(sys.argv) > 1 else ""
scripts = sorted(p for p in HERE.glob("T2*.py") if p.name.startswith(prefix))
env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")   # figures saved, never shown; any console encoding
for p in scripts:
    t0 = time.time()
    print(f"\n##### {p.name}", flush=True)
    r = subprocess.run([sys.executable, "-W", "ignore", str(p)], cwd=HERE, env=env)
    if r.returncode:
        sys.exit(f"{p.name} failed ({r.returncode})")
    print(f"##### {p.name} done in {time.time() - t0:.0f} s", flush=True)
print(f"\n{len(scripts)} ask scripts ran without error")
