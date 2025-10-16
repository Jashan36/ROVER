#!/usr/bin/env python3
"""Serve an IP camera stream to dashboard clients via WebSocket."""

import argparse
import asyncio
import base64
import json
import logging
import time

import cv2
import websockets


logger = logging.getLogger(__name__)


async def frame_generator(stream_url: str, max_fps: float):
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera stream: {stream_url}")

    frame_interval = 1.0 / max(max_fps, 0.01)
    last_sent = 0.0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            await asyncio.sleep(0.1)
            continue

        now = time.time()
        if now - last_sent < frame_interval:
            await asyncio.sleep(0.01)
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        success, buf = cv2.imencode(".jpg", frame_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not success:
            logger.warning("Failed to encode frame")
            continue

        image_b64 = base64.b64encode(buf).decode("utf-8")

        payload = {
            "timestamp": now,
            "overlay_image": f"data:image/jpeg;base64,{image_b64}",
            "avg_traversability": 0.0,
            "safe_area_ratio": 0.0,
            "best_direction_deg": 0.0,
            "num_hazards": 0,
            "hazard_summary": {},
            "fps": 0.0,
            "terrain_distribution": {},
        }

        last_sent = now
        yield json.dumps(payload)


async def client_handler(websocket, stream_url: str, max_fps: float):
    logger.info("Dashboard connected from %s", websocket.remote_address)
    async for payload in frame_generator(stream_url, max_fps):
        await websocket.send(payload)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Serve an IP camera stream to dashboard clients via WebSocket."
    )
    parser.add_argument("stream_url", help="Camera URL, e.g. http://phone-ip:8080/video")
    parser.add_argument("--host", default="0.0.0.0", help="WebSocket host (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket port (default 8765)")
    parser.add_argument("--max-fps", type=float, default=5.0, help="Maximum frames per second")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


async def async_main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    async def handler(websocket, path=None):
        await client_handler(websocket, args.stream_url, args.max_fps)

    async with websockets.serve(handler, args.host, args.port):
        logger.info("Serving WebSocket on ws://%s:%d", args.host, args.port)
        await asyncio.Future()  # run forever


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
