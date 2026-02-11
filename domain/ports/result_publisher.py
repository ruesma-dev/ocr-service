# domain/ports/result_publisher.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class ResultPublisher(ABC):
    @abstractmethod
    def publish(self, payload: Dict[str, Any]) -> None:
        raise NotImplementedError
