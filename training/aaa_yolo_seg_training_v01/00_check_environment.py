#!/usr/bin/env python3
from __future__ import annotations

import platform

import torch
import ultralytics


print("python:", platform.python_version())
print("torch:", torch.__version__)
print("ultralytics:", ultralytics.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda_device_0:", torch.cuda.get_device_name(0))
else:
    raise SystemExit("CUDA is unavailable. Run training in a GPU-enabled environment.")

