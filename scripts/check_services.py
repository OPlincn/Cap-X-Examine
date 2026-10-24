#!/usr/bin/env python3
"""Pre-episode health check for the shared perception/IK services (exit 0 = healthy)."""

from __future__ import annotations

import socket
import sys
import time

import numpy as np


def _port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(3.0)
        return s.connect_ex((host, port)) == 0


def main() -> int:
    for name, port in (("sam3", 8114), ("graspnet", 8115), ("pyroki", 8116)):
        if not _port_open(port):
            print(f"[health] {name} :{port} not accepting connections")
            return 1
    from capx.integrations.motion.pyroki import init_pyroki

    t = time.time()
    try:
        q = init_pyroki()(np.array([0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.3]))
    except Exception as exc:  # noqa: BLE001
        print(f"[health] pyroki /ik failed: {exc}")
        return 1
    if q.shape != (7,) and q.shape[-1] < 7:
        print(f"[health] pyroki /ik returned unexpected shape {q.shape}")
        return 1
    print(f"[health] ok (ik {time.time() - t:.2f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
