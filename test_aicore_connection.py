"""
Test connection to SAP AI Core using EnergyForecast.json service key.
Lists available resource groups, scenarios, and deployments.

Usage:
  python3 test_aicore_connection.py

Requires:
  pip3 install ai-core-sdk
  EnergyForecast.json in the same directory as this script.
"""

import json
import os
import sys

try:
    from ai_core_sdk.ai_core_v2_client import AICoreV2Client
except ImportError:
    print("ERROR: ai-core-sdk not installed. Run: pip3 install ai-core-sdk")
    sys.exit(1)

# Load service key (same directory as script, or override path here)
SK_PATH = os.path.join(os.path.dirname(__file__), "EnergyForecast.json")

if not os.path.exists(SK_PATH):
    print(f"ERROR: Service key not found at {SK_PATH}")
    sys.exit(1)

with open(SK_PATH) as f:
    sk = json.load(f)

# Build client
client = AICoreV2Client(
    base_url=sk["serviceurls"]["AI_API_URL"] + "/v2",
    auth_url=sk["url"] + "/oauth/token",
    client_id=sk["clientid"],
    client_secret=sk["clientsecret"],
)

print("=" * 60)
print("SAP AI Core — Connection Test")
print("=" * 60)
print(f"API URL : {sk['serviceurls']['AI_API_URL']}")
print(f"Auth URL: {sk['url']}")
print(f"Zone    : {sk['identityzone']}")
print()

# --- Resource Groups ---
print("[ Resource Groups ]")
try:
    rgs = client.resource_groups.query()
    items = getattr(rgs, "resources", [])
    if items:
        for rg in items:
            print(f"  - {rg.resource_group_id}")
    else:
        print("  (none found)")
except Exception as e:
    print(f"  ERROR: {e}")

# Use default resource group for subsequent queries
rg_id = "default"

print()
print("[ Scenarios ]")
try:
    scenarios = client.scenario.query(resource_group=rg_id)
    items = getattr(scenarios, "resources", [])
    if items:
        for s in items:
            print(f"  - {s.id} | {getattr(s, 'name', '')}")
    else:
        print("  (none found in 'default' resource group)")
except Exception as e:
    print(f"  ERROR: {e}")

print()
print("[ Deployments ]")
try:
    deployments = client.deployment.query(resource_group=rg_id)
    items = getattr(deployments, "resources", [])
    if items:
        for d in items:
            print(f"  - {d.id} | status={d.status} | scenario={getattr(d, 'scenario_id', 'N/A')}")
    else:
        print("  (none found in 'default' resource group)")
except Exception as e:
    print(f"  ERROR: {e}")

print()
print("Connection test complete.")
