import os
import sys

# macOS: PyTorch and XGBoost each ship their own OpenMP runtime; loading both in one process
# deadlocks on the first BERT encode (the server stays up but never answers). One OpenMP thread
# avoids it. Must be set before torch/xgboost are imported. Override by setting it yourself.
if sys.platform == 'darwin':
    os.environ.setdefault('OMP_NUM_THREADS', '1')
