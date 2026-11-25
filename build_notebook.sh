#!/usr/bin/env bash
# jupytext 원본(.py) → 실행된 노트북(.ipynb) + 웹 페이지용 데이터(docs/*.js)
# 사용법: ./build_notebook.sh            (둘 다)
#         ./build_notebook.sh etf_map    (하나만)
set -euo pipefail
cd "$(dirname "$0")"
for name in "${@:-etf_map volatility_forecast}"; do
  for nb in $name; do
    jupytext --to ipynb "$nb.py" -o "$nb.ipynb"
    jupyter nbconvert --to notebook --execute --inplace "$nb.ipynb" --ExecutePreprocessor.timeout=1800
    # 텐서플로가 stderr로 내는 로그는 노트북에서 지운다
    python3 - "$nb.ipynb" <<'PY'
import json, sys
p = sys.argv[1]
nb = json.load(open(p, encoding="utf-8"))
for c in nb["cells"]:
    if c["cell_type"] == "code":
        c["outputs"] = [o for o in c["outputs"] if not (o["output_type"] == "stream" and o.get("name") == "stderr")]
json.dump(nb, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
PY
  done
done
