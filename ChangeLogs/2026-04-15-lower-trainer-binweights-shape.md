# Lower Trainer Bin Weight Shape Fix

## Changes
- Made `train_Lower_MELTS()` build a 2D default bin-weight row from
  `ml_indexer.nphases` when no tensor is passed.
- Normalized 1D bin-weight inputs into a row tensor before loss and
  metric code uses `shape[1]`.