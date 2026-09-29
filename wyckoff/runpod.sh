#!/usr/bin/env bash
# Train the range model on a RunPod GPU pod (any PyTorch template).
#   1. Start a pod with a PyTorch image, open its terminal.
#   2. git clone https://github.com/NiftyJ/smc_dickson_type_order && cd smc_dickson_type_order
#   3. Upload your MT5 export and your labels (my_ranges.json) into data/
#   4. bash wyckoff/runpod.sh data/V75_M15.csv data/my_ranges.json
# The results print at the end; the model is saved to wyckoff/range_model.pt and example
# pictures to outputs/wyckoff/. Download those, or send me the printed results.
set -euo pipefail
DATA="${1:?usage: bash wyckoff/runpod.sh <prices.csv> [labels.json] [epochs]}"
LABELS="${2:-}"
EPOCHS="${3:-20}"
pip install -q numpy pandas scipy scikit-learn matplotlib lightgbm
python -c "import torch; print('GPU:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
if [ -n "$LABELS" ]; then
  python -m wyckoff.train --data "$DATA" --labels "$LABELS" --epochs "$EPOCHS" --stride 1
else
  python -m wyckoff.train --data "$DATA" --epochs "$EPOCHS" --stride 1
fi
