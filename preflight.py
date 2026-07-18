"""
preflight.py
------------
Run this before ./run_all.sh to verify the environment is ready.
Checks Python version, .env keys, package imports, EIA API, TabPFN API,
disk space, and Apple Silicon MPS availability.

Usage:
    python preflight.py
"""

import sys
import os
import shutil
from pathlib import Path

# ── helpers ───────────────────────────────────────────────────────────────────

PASS  = "\033[92m  PASS\033[0m"
FAIL  = "\033[91m  FAIL\033[0m"
WARN  = "\033[93m  WARN\033[0m"
SKIP  = "\033[90m  SKIP\033[0m"
HEAD  = "\033[1m"
RESET = "\033[0m"

failures = []
warnings = []

def ok(label, detail=""):
    print(f"{PASS}  {label}" + (f"  ({detail})" if detail else ""))

def fail(label, detail="", fatal=True):
    print(f"{FAIL}  {label}" + (f"  → {detail}" if detail else ""))
    if fatal:
        failures.append(label)

def warn(label, detail=""):
    print(f"{WARN}  {label}" + (f"  → {detail}" if detail else ""))
    warnings.append(label)

def skip(label, detail=""):
    print(f"{SKIP}  {label}" + (f"  ({detail})" if detail else ""))

def section(title):
    print(f"\n{HEAD}{title}{RESET}")
    print("─" * 50)


# ── load .env ─────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent
env_path = ROOT / ".env"

try:
    from dotenv import load_dotenv
    load_dotenv(env_path)
except ImportError:
    pass  # will be caught in imports check


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Python version
# ═══════════════════════════════════════════════════════════════════════════════

section("1. Python version")

major, minor = sys.version_info[:2]
ver_str = f"{major}.{minor}.{sys.version_info[2]}"

if major == 3 and minor >= 10:
    ok("Python version", ver_str)
elif major == 3 and minor == 9:
    warn("Python 3.9 detected — tabpfn requires >=3.10", f"found {ver_str}")
else:
    fail("Python version too old — need >=3.10", f"found {ver_str}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. .env file and keys
# ═══════════════════════════════════════════════════════════════════════════════

section("2. Environment / .env")

if env_path.exists():
    ok(".env file found", str(env_path))
else:
    fail(".env file missing",
         "run: cp .env.example .env")

eia_key = os.getenv("EIA_API_KEY", "")
if eia_key and not eia_key.startswith("your_"):
    ok("EIA_API_KEY set", f"{eia_key[:8]}…")
else:
    fail("EIA_API_KEY not set or is placeholder",
         "add real key to .env")

tabpfn_key = os.getenv("TABPFN_API_KEY", "")
if tabpfn_key and tabpfn_key.startswith("tabpfn_sk_"):
    ok("TABPFN_API_KEY set", f"{tabpfn_key[:18]}…")
elif tabpfn_key:
    warn("TABPFN_API_KEY set but format looks unexpected",
         "expected tabpfn_sk_… prefix")
else:
    warn("TABPFN_API_KEY not set",
         "will use local OSS TabPFN-3 — OK for portfolio use")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Required Python packages
# ═══════════════════════════════════════════════════════════════════════════════

section("3. Package imports")

packages = [
    ("pandas",       "pandas"),
    ("numpy",        "numpy"),
    ("sklearn",      "scikit-learn"),
    ("xgboost",      "xgboost"),
    ("statsmodels",  "statsmodels"),
    ("mlflow",       "mlflow"),
    ("fastapi",      "fastapi"),
    ("uvicorn",      "uvicorn"),
    ("pydantic",     "pydantic"),
    ("matplotlib",   "matplotlib"),
    ("dotenv",       "python-dotenv"),
    ("requests",     "requests"),
    ("tabpfn",       "tabpfn"),
]

for mod, pip_name in packages:
    try:
        m = __import__(mod)
        ver = getattr(m, "__version__", "?")
        ok(pip_name, ver)
    except ImportError:
        fail(f"{pip_name} not installed",
             f"run: pip install {pip_name}")

# tabpfn-client is optional
try:
    import tabpfn_client
    ok("tabpfn-client (optional)", getattr(tabpfn_client, "__version__", "?"))
except ImportError:
    if tabpfn_key:
        fail("tabpfn-client not installed but TABPFN_API_KEY is set",
             "run: pip install tabpfn-client")
    else:
        skip("tabpfn-client (not installed, no API key set — OK)")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. macOS / XGBoost OpenMP check
# ═══════════════════════════════════════════════════════════════════════════════

section("4. macOS OpenMP (XGBoost)")

if sys.platform == "darwin":
    libomp_paths = [
        "/usr/local/lib/libomp.dylib",
        "/opt/homebrew/lib/libomp.dylib",
        "/usr/local/opt/libomp/lib/libomp.dylib",
    ]
    found_libomp = any(Path(p).exists() for p in libomp_paths)
    if found_libomp:
        found_path = next(p for p in libomp_paths if Path(p).exists())
        ok("libomp found", found_path)
    else:
        fail("libomp not found — XGBoost will crash on Mac",
             "fix: brew install libomp && sudo ln -s $(brew --prefix libomp)/lib/libomp.dylib /usr/local/lib/libomp.dylib")

    # Quick XGBoost smoke test
    try:
        import xgboost as xgb
        import numpy as np
        m = xgb.XGBRegressor(n_estimators=2, verbosity=0)
        m.fit(np.array([[1],[2],[3]]), np.array([1,2,3]))
        ok("XGBoost smoke test passed")
    except Exception as e:
        fail("XGBoost smoke test failed", str(e)[:80])
else:
    skip("macOS OpenMP check (not on macOS)")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. GPU / Apple Silicon MPS
# ═══════════════════════════════════════════════════════════════════════════════

section("5. GPU / accelerator")

try:
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        ok("CUDA GPU available", name)
    elif sys.platform == "darwin" and torch.backends.mps.is_available():
        ok("Apple Silicon MPS available",
           f"MPS fraction = {os.getenv('TABPFN_MPS_MEMORY_FRACTION','not set')}")
    else:
        warn("No GPU detected — TabPFN-3 will run on CPU",
             "expect slower inference; still works for this dataset size")
except ImportError:
    warn("PyTorch not importable directly — tabpfn will install it",
         "first run may be slow while dependencies resolve")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. EIA API connectivity
# ═══════════════════════════════════════════════════════════════════════════════

section("6. EIA API connectivity")

if eia_key and not eia_key.startswith("your_"):
    try:
        import requests
        url = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
        params = {
            "api_key":              eia_key,
            "frequency":            "hourly",
            "data[]":               "value",
            "facets[respondent][]": "PJM",
            "facets[type][]":       "D",
            "length":               1,
        }
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json().get("response", {}).get("data", [])
            if data:
                ok("EIA API reachable", f"latest record: {data[0].get('period','?')}")
            else:
                warn("EIA API responded but returned no data",
                     "check key or try again shortly")
        elif r.status_code == 403:
            fail("EIA API key rejected (403)", "check EIA_API_KEY in .env")
        else:
            warn(f"EIA API returned HTTP {r.status_code}", r.text[:80])
    except requests.exceptions.ConnectionError:
        fail("Cannot reach api.eia.gov", "check internet connection")
    except Exception as e:
        warn("EIA API check failed", str(e)[:80])
else:
    skip("EIA API check (key not set)")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. TabPFN API connectivity
# ═══════════════════════════════════════════════════════════════════════════════

section("7. TabPFN API connectivity")

if tabpfn_key and tabpfn_key.startswith("tabpfn_sk_"):
    try:
        import requests
        r = requests.get(
            "https://api.priorlabs.ai/health",
            headers={"Authorization": f"Bearer {tabpfn_key}"},
            timeout=10,
        )
        if r.status_code == 200:
            ok("TabPFN API reachable and key accepted")
        elif r.status_code == 401:
            fail("TabPFN API key rejected (401)",
                 "check TABPFN_API_KEY in .env")
        else:
            warn(f"TabPFN API returned HTTP {r.status_code}",
                 "may still work — Prior Labs health endpoint path may differ")
    except requests.exceptions.ConnectionError:
        fail("Cannot reach Prior Labs API", "check internet connection")
    except Exception as e:
        warn("TabPFN API check inconclusive", str(e)[:80])
else:
    skip("TabPFN API check (no API key set — will use local OSS)")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Disk space
# ═══════════════════════════════════════════════════════════════════════════════

section("8. Disk space")

total, used, free = shutil.disk_usage(ROOT)
free_gb = free / (1024 ** 3)

# TabPFN checkpoint ~500MB, data ~100MB, models ~50MB → need ~1GB comfortable
if free_gb >= 2.0:
    ok("Disk space", f"{free_gb:.1f} GB free")
elif free_gb >= 0.8:
    warn("Low disk space", f"{free_gb:.1f} GB free — TabPFN checkpoint is ~500 MB")
else:
    fail("Insufficient disk space", f"{free_gb:.1f} GB free — need at least 1 GB")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Project structure
# ═══════════════════════════════════════════════════════════════════════════════

section("9. Project structure")

expected = [
    "src/data_pipeline_eia.py",
    "src/train_xgboost.py",
    "src/train_sarima.py",
    "src/train_tabpfn.py",
    "src/train_intervals.py",
    "src/backtest.py",
    "src/evaluate.py",
    "src/retrain.py",
    "src/features.py",
    "src/tracker.py",
    "api/main.py",
    "frontend/index.html",
    "run_all.sh",
    "requirements.txt",
]

for rel in expected:
    p = ROOT / rel
    if p.exists() and p.stat().st_size > 100:
        ok(rel)
    elif p.exists():
        fail(f"{rel} exists but looks like a stub", f"{p.stat().st_size} bytes")
    else:
        fail(f"{rel} missing")


# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'═'*50}")
print(f"{HEAD}PREFLIGHT SUMMARY{RESET}")
print(f"{'═'*50}")

if not failures and not warnings:
    print(f"\033[92m  All checks passed. Ready to run ./run_all.sh\033[0m\n")
elif not failures:
    print(f"\033[93m  {len(warnings)} warning(s), 0 failures.\033[0m")
    for w in warnings:
        print(f"    ⚠  {w}")
    print(f"\n\033[93m  Likely OK to proceed — review warnings above.\033[0m\n")
else:
    print(f"\033[91m  {len(failures)} failure(s), {len(warnings)} warning(s).\033[0m")
    print(f"\033[91m  Fix failures before running ./run_all.sh:\033[0m")
    for f in failures:
        print(f"    ✗  {f}")
    if warnings:
        print(f"\033[93m  Warnings:\033[0m")
        for w in warnings:
            print(f"    ⚠  {w}")
    print()
    sys.exit(1)
