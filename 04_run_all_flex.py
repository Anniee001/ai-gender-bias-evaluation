# 04_run_all_flex.py
import sys, subprocess

PIPELINE = [
    [sys.executable, "01_explicit_bias_eval_flex.py"],
    [sys.executable, "02_implicit_bias_eval_flex.py"],
    [sys.executable, "03_network_analysis_flex.py"],
]

for cmd in PIPELINE:
    print("\n>>> RUN:", " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"❌ 出错：{' '.join(cmd)}")

print("\n✅ 全部完成（01→02→03）")
