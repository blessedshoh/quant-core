"""The fixed income lab's routes: bondlab's lab service behind the existing API.

All the work happens in bondlab (``bondlab.ui.service.handle``), which turns a JSON
payload into the engine's explicit objects and back and computes nothing itself. This
router only checks the request's size and shape, runs the call off the event loop, and
maps bondlab's named errors to HTTP 422. bondlab bounds every number a route reads, so no
single request can hold the shared API's CPU for long.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from bondlab.ui.payloads import PayloadError
from bondlab.ui.service import handle

ROUTES = ("treasury-curve", "bond", "hedge", "basics", "rate-tree", "convertible")
MAX_BODY_BYTES = 200_000

router = APIRouter(prefix="/fixed-income", tags=["fixed income"])


def _too_large() -> HTTPException:
    return HTTPException(status_code=413, detail=f"request body over {MAX_BODY_BYTES:,} bytes")


async def _capped_body(request: Request) -> bytes:
    """Read the body, refusing it from its declared length or as soon as it passes the cap."""
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > MAX_BODY_BYTES:
        raise _too_large()
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > MAX_BODY_BYTES:
            raise _too_large()
    return bytes(body)


@router.post("/{route}")
async def fixed_income(route: str, request: Request) -> dict:
    """Run one fixed income lab route. Rates in percent, spreads in basis points."""
    if route not in ROUTES:
        raise HTTPException(status_code=404, detail=f"unknown route {route!r}; this API serves: {', '.join(ROUTES)}")
    body = await _capped_body(request)
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=422, detail=f"request body is not JSON: {error}") from error
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="request body must be a JSON object")
    try:
        result = await run_in_threadpool(handle, route, payload)
    except (PayloadError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return dict(result)
