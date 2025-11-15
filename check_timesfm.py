#!/usr/bin/env python3
"""Check TimesFM API."""

import timesfm
import inspect

print("=== TimesFM Package Inspection ===\n")

# Check available attributes
print("Available attributes:")
attrs = [x for x in dir(timesfm) if not x.startswith('_')]
for attr in attrs:
    print(f"  - {attr}")

print("\n=== Class Checks ===")
print(f"Has TimesFm: {hasattr(timesfm, 'TimesFm')}")
print(f"Has TimesFmTorch: {hasattr(timesfm, 'TimesFmTorch')}")
print(f"Has TimesFmCheckpoint: {hasattr(timesfm, 'TimesFmCheckpoint')}")

# Check if TimesFm exists and its methods
if hasattr(timesfm, 'TimesFm'):
    print("\n=== TimesFm class methods ===")
    methods = [x for x in dir(timesfm.TimesFm) if not x.startswith('_')]
    for method in methods:
        print(f"  - {method}")

    # Check signature of __init__
    print("\n=== TimesFm.__init__ signature ===")
    try:
        sig = inspect.signature(timesfm.TimesFm.__init__)
        print(f"  {sig}")
    except Exception as e:
        print(f"  Error: {e}")

# Check for checkpoint loading functions
print("\n=== Looking for checkpoint/pretrained methods ===")
for attr in attrs:
    obj = getattr(timesfm, attr)
    if callable(obj) or inspect.isclass(obj):
        if hasattr(obj, 'from_checkpoint') or hasattr(obj, 'from_pretrained'):
            print(f"  {attr} has from_checkpoint or from_pretrained")
            if hasattr(obj, 'from_checkpoint'):
                print(f"    - from_checkpoint signature: {inspect.signature(obj.from_checkpoint)}")
            if hasattr(obj, 'from_pretrained'):
                print(f"    - from_pretrained signature: {inspect.signature(obj.from_pretrained)}")
