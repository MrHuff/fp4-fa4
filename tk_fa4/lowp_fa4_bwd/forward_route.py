"""Process-global forward-route activation with redundant-write elision."""

from __future__ import annotations

import os


FORWARD_ROUTE_ENV = "TK_FA4_FP4PV_FWD_CONFIG"
_active_forward_route: str | None = None


def activate_forward_route(route: str) -> bool:
    """Activate ``route`` and return whether the environment was updated.

    A bound model owns the route token.  Checking the environment here keeps
    model-boundary activation safe if a legacy caller mutates it directly;
    layers use :func:`require_active_forward_route`, which has no environment
    lookup on the timed path.
    """
    if not isinstance(route, str) or not route:
        raise ValueError("forward route must be a non-empty string")
    global _active_forward_route
    if (
        _active_forward_route == route
        and os.environ.get(FORWARD_ROUTE_ENV) == route
    ):
        return False
    os.environ[FORWARD_ROUTE_ENV] = route
    _active_forward_route = route
    return True


def require_active_forward_route(route: str) -> None:
    """Reject a layer whose runtime differs from its model-bound route."""
    if _active_forward_route != route:
        raise RuntimeError(
            "low-precision attention runtime is not bound at the model "
            f"boundary: active={_active_forward_route!r}, requested={route!r}"
        )
