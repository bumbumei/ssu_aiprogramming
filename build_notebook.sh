#!/usr/bin/env bash
# volatility_forecast.py (jupytext 원본) → 실행된 volatility_forecast.ipynb + docs/data.js
set -euo pipefail
cd "$(dirname "$0")"
jupytext --to ipynb volatility_forecast.py -o volatility_forecast.ipynb
jupyter nbconvert --to notebook --execute --inplace volatility_forecast.ipynb --ExecutePreprocessor.timeout=1800
# 텐서플로가 stderr로 내는 로그는 노트북에서 지운다
python3 - <<'PY'
import json
p = "volatility_forecast.ipynb"
nb = json.load(open(p, encoding="utf-8"))
for c in nb["cells"]:
    if c["cell_type"] == "code":
        c["outputs"] = [o for o in c["outputs"] if not (o["output_type"] == "stream" and o.get("name") == "stderr")]
json.dump(nb, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
PY
