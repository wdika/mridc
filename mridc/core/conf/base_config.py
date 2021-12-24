# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

from dataclasses import dataclass
from typing import Optional

__all__ = ["Config"]


@dataclass
class Config:
    """Abstract mridc Configuration class."""

    name: Optional[str] = None
