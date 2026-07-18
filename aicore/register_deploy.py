"""
aicore/register_deploy.py
--------------------------
Registers the energy forecasting model as a deployment on SAP AI Core.

Steps performed:
  1. Connect to AI Core using EnergyForecast.json service key
  2. Register Docker Hub credentials as a registry secret
  3. Register GitHub repo as an AI Core Application (git-ops sync)
  4. Wait for the scenario + executable to sync from GitHub
  5. Create a deployment configuration
  6. Launch the deployment
  7. Poll until deployment is RUNNING and print the inference URL

Prerequisites:
  - Docker image built and pushed:
      docker build -f aicore/Dockerfile -t YOUR_DOCKERHUB_USERNAME/energy-forecast-server:v1 .
      docker push YOUR_DOCKERHUB_USERNAME/energy-forecast-server:v1
  - serving_template.yaml committed and pushed to GitHub
  - EnergyForecast.json in the project root

Usage:
  python3 aicore/register_deploy.py

Configure the four constants below before running.
"""

import json
import time
import base64
from pathlib import Path

from ai_core_sdk.ai_core_v2_client import AICoreV2Client

# ─────────────────────────────────────────────
# CONFIGURE THESE BEFORE RUNNING
# ─────────────────────────────────────────────

DOCKERHUB_USERNAME = "rali24"
DOCKERHUB_PASSWORD = "YOUR_DOCKERHUB_PASSWORD"   # Docker Hub password or access token

GITHUB_REPO_URL    = "https://github.com/I567946/energy-forecast"
GITHUB_USERNAME    = "YOUR_GITHUB_USERNAME"
GITHUB_TOKEN       = "YOUR_GITHUB_TOKEN"          # Personal access token (repo read scope)

RESOURCE_GROUP     = "predictionmodel"            # AI Core resource group to deploy into

# ─────────────────────────────────────────────

SK_PATH = Path(__file__).parent.parent / "EnergyForecast.json"


def connect() -> AICoreV2Client:
    with open(SK_PATH) as f:
        sk = json.load(f)
    client = AICoreV2Client(
        base_url=sk["serviceurls"]["AI_API_URL"] + "/v2",
        auth_url=sk["url"] + "/oauth/token",
        client_id=sk["clientid"],
        client_secret=sk["clientsecret"],
    )
    print("[1/6] Connected to SAP AI Core.")
    return client


def register_docker_secret(client: AICoreV2Client):
    """Register Docker Hub credentials so AI Core can pull the image."""
    # Docker config JSON — same format as ~/.docker/config.json
    auth_str  = base64.b64encode(f"{DOCKERHUB_USERNAME}:{DOCKERHUB_PASSWORD}".encode()).decode()
    docker_cfg = json.dumps({
        "auths": {
            "https://index.docker.io/v1/": {"auth": auth_str}
        }
    })

    try:
        client.docker_registry_secrets.create(
            name="docker-registry-secret",
            data={".dockerconfigjson": docker_cfg},
        )
        print("[2/6] Docker registry secret created.")
    except Exception as e:
        if "already exists" in str(e).lower() or "conflict" in str(e).lower():
            print("[2/6] Docker registry secret already exists — skipping.")
        else:
            raise


def register_github_app(client: AICoreV2Client):
    """
    Register the GitHub repo as an AI Core Application.
    AI Core will sync the aicore/serving_template.yaml from this repo
    and automatically create the scenario + executable.
    """
    try:
        client.applications.create(
            application_name="energy-forecast-app",
            repository_url=GITHUB_REPO_URL,
            path="aicore",                  # subfolder containing serving_template.yaml
            revision="HEAD",
            username=GITHUB_USERNAME,
            password=GITHUB_TOKEN,
        )
        print("[3/6] GitHub application registered.")
    except Exception as e:
        if "already exists" in str(e).lower() or "conflict" in str(e).lower():
            print("[3/6] Application already registered — skipping.")
        else:
            raise


def wait_for_executable(client: AICoreV2Client,
                         scenario_id: str = "energy-forecast",
                         executable_id: str = "energy-forecast-server",
                         timeout: int = 120) -> bool:
    """Poll until AI Core has synced the scenario/executable from GitHub."""
    print(f"[4/6] Waiting for executable '{executable_id}' to sync from GitHub", end="", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        try:
            execs = client.executable.query(
                scenario_id=scenario_id,
                resource_group=RESOURCE_GROUP,
            )
            ids = [e.id for e in getattr(execs, "resources", [])]
            if executable_id in ids:
                print(" done.")
                return True
        except Exception:
            pass
        print(".", end="", flush=True)
        time.sleep(10)
    print(" timed out.")
    return False


def create_deployment(client: AICoreV2Client) -> str:
    """Create a configuration and launch the deployment."""
    scenario_id   = "energy-forecast"
    executable_id = "energy-forecast-server"

    # Create configuration
    config = client.configuration.create(
        name="energy-forecast-config-v1",
        scenario_id=scenario_id,
        executable_id=executable_id,
        resource_group=RESOURCE_GROUP,
    )
    config_id = config.id
    print(f"[5/6] Configuration created: {config_id}")

    # Create deployment
    deployment = client.deployment.create(
        configuration_id=config_id,
        resource_group=RESOURCE_GROUP,
    )
    deployment_id = deployment.id
    print(f"[6/6] Deployment launched: {deployment_id}")
    return deployment_id


def wait_for_deployment(client: AICoreV2Client,
                         deployment_id: str,
                         timeout: int = 600) -> str | None:
    """Poll until deployment status is RUNNING and return the inference URL."""
    print(f"\nWaiting for deployment {deployment_id} to reach RUNNING status", end="", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        d = client.deployment.get(
            deployment_id=deployment_id,
            resource_group=RESOURCE_GROUP,
        )
        status = getattr(d, "status", "UNKNOWN")
        if status == "RUNNING":
            url = getattr(d, "deployment_url", None)
            print(f"\nDeployment is RUNNING.")
            print(f"\nInference URL: {url}")
            print(f"\nTest with:")
            print(f'  curl -X POST "{url}/v2/predict" \\')
            print(f'    -H "Content-Type: application/json" \\')
            print(f'    -d \'{{"hours": 24}}\'')
            return url
        elif status in ("DEAD", "STOPPED", "ERROR"):
            print(f"\nDeployment failed with status: {status}")
            return None
        print(".", end="", flush=True)
        time.sleep(15)
    print("\nTimed out waiting for deployment.")
    return None


def main():
    # Validate config
    for var, val in [
        ("DOCKERHUB_USERNAME", DOCKERHUB_USERNAME),
        ("DOCKERHUB_PASSWORD", DOCKERHUB_PASSWORD),
        ("GITHUB_REPO_URL",    GITHUB_REPO_URL),
        ("GITHUB_USERNAME",    GITHUB_USERNAME),
        ("GITHUB_TOKEN",       GITHUB_TOKEN),
    ]:
        if "YOUR_" in val:
            print(f"ERROR: Please set {var} at the top of this file before running.")
            return

    client        = connect()
    register_docker_secret(client)
    register_github_app(client)
    synced        = wait_for_executable(client)
    if not synced:
        print("Executable did not sync in time. Check AI Core application status.")
        print("Tip: In AI Core cockpit, check Applications > energy-forecast-app for sync errors.")
        return
    deployment_id = create_deployment(client)
    wait_for_deployment(client, deployment_id)


if __name__ == "__main__":
    main()
