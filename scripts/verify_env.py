import importlib

pkgs = ["numpy", "pandas", "PIL", "sklearn", "torch", "torchvision", "matplotlib", "pyarrow", "tqdm"]
for name in pkgs:
    try:
        importlib.import_module(name)
        print(f"[OK] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {e}")
