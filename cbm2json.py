#!/usr/bin/env python3
"""
Convert a CatBoost binary model (.cbm) to gzip-compressed JSON (.json.gz).

Usage:
  python3 cbm2json.py model.cbm
  python3 cbm2json.py model.cbm output.json.gz
"""

import gzip
import os
import sys

try:
    import catboost as cb
except ImportError:
    print("catboost not found. Install with: pip install catboost", file=sys.stderr)
    sys.exit(1)


def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) == 3 else os.path.splitext(src)[0] + ".json.gz"

    model = cb.CatBoost()
    model.load_model(src)

    tmp = dst + ".tmp"
    model.save_model(tmp, format="json")
    with open(tmp, "rb") as f_in, gzip.open(dst, "wb", compresslevel=9) as f_out:
        f_out.write(f_in.read())
    os.remove(tmp)

    print(f"{src} -> {dst}")


if __name__ == "__main__":
    main()
