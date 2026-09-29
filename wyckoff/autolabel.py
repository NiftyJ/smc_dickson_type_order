"""
Starter boxes for the labelling page.

    python -m wyckoff.autolabel --data "data/V75_M15.csv"
writes starter_ranges.json next to your CSV. Open wyckoff/label_tool.html in a browser,
load the CSV and this file, fix the boxes, and export my_ranges.json.
"""
import argparse
import os

import config as cfg
from smcml.data import load_mt5_csv
from wyckoff.ranges import find_ranges, save_boxes

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    df = load_mt5_csv(args.data)
    boxes = find_ranges(df, cfg)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.data)), "starter_ranges.json")
    save_boxes(boxes, out, cfg.RANGE_TF)
    print(f"{len(boxes)} starter ranges on {cfg.RANGE_TF} written to {out}")
