# infrastructure/publishers/http_webhook_publisher.py
from __future__ import annotations

import logging
from typing import Any, Dict

import httpx

from domain.ports.result_publisher import ResultPublisher

logger = logging.getLogger(__name__)


class HttpWebhookPublisher(ResultPublisher):
    def __init__(self, url: str, timeout_s: int) -> None:
        self._url = url
        self._timeout = timeout_s
        self._client = httpx.Client(timeout=self._timeout)

    def publish(self, payload: Dict[str, Any]) -> None:
        logger.info("Webhook POST -> %s", self._url)
        resp = self._client.post(self._url, json=payload)
        if resp.status_code < 200 or resp.status_code >= 300:
            raise RuntimeError(f"Webhook error {resp.status_code}: {resp.text[:300]}")
        logger.info("Webhook OK (%s)", resp.status_code)
