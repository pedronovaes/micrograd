"""
Wrapper to pytorch-like tensor object.
"""

from __future__ import annotations
from typing import Tuple
import math


class Value:
    """
    TO DO.
    """

    def __init__(self, data: int | float, _children: Tuple = ()) -> None:
        self.data = data
        self.grad = 0.0

        # Attribute used to build autograd graph.
        self._previous = set(_children)
        self._backward = lambda: None  # To do a little piece of chain rule.

    def __repr__(self) -> str:
        return f'Value(data={self.data}, grad={self.grad})'
