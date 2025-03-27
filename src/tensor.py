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

    def __add__(self, other: Value) -> Value:
        output = Value(data=self.data + other.data, _children=(self, other))

        return output

    def __mul__(self, other: Value) -> Value:
        output = Value(data=self.data * other.data, _children=(self, other))

        return output

    # pylint: disable=C0116
    def tanh(self) -> Value:
        tan = (math.exp(2 * self.data) - 1) / (math.exp(2 * self.data) + 1)
        output = Value(data=tan, _children=(self,))

        return output
