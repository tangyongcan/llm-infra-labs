"""Load generator: admits synthetic requests to the broker.

Two shapes, because they stress the autoscaler differently:
  burst - dump everything at once (a traffic spike; worst case for cold start)
  ramp  - hold a steady arrival rate for a while (a sustained load increase)

Stdlib only, so it runs in the k8s tooling env without extra installs.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request


def post(url: str, params: dict) -> dict:
    full = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(full, method="POST", data=b"")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:18000")
    ap.add_argument("--shape", choices=["burst", "ramp"], default="burst")
    ap.add_argument("--total", type=int, default=120, help="requests to admit")
    ap.add_argument(
        "--work-ms", type=int, default=1500, help="simulated service time"
    )
    ap.add_argument(
        "--rate", type=float, default=8.0, help="requests/s for ramp shape"
    )
    ap.add_argument("--reset", action="store_true", help="reset broker first")
    args = ap.parse_args()

    if args.reset:
        post(f"{args.base}/reset", {})
        print("broker reset")

    t0 = time.time()
    if args.shape == "burst":
        r = post(
            f"{args.base}/enqueue",
            {"n": args.total, "work_ms": args.work_ms},
        )
        print(
            f"t=+0.0s burst admitted {args.total} requests, "
            f"queue_depth={r['queue_depth']}"
        )
    else:
        admitted = 0
        interval = 1.0 / args.rate
        while admitted < args.total:
            r = post(f"{args.base}/enqueue", {"n": 1, "work_ms": args.work_ms})
            admitted += 1
            if admitted % 10 == 0:
                print(
                    f"t=+{time.time() - t0:5.1f}s admitted {admitted}"
                    f"/{args.total} queue_depth={r['queue_depth']}"
                )
            time.sleep(interval)

    print(f"load generation done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
