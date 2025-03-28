"""
Wrapper to pytorch-like tensor object.
"""

from __future__ import annotations
from typing import Tuple
import math

from src.utils import build_topological_order


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

        # Compute gradients. An add operator is just a distributor of
        # gradients (the children nodes receives the same father's gradient).
        def _backward():
            self.grad += output.grad
            other.grad += output.grad

        output._backward = _backward

        return output

    def __mul__(self, other: Value) -> Value:
        output = Value(data=self.data * other.data, _children=(self, other))

        def _backward():
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad

        output._backward = _backward

        return output

    # pylint: disable=C0116
    def tanh(self) -> Value:
        tan = (math.exp(2 * self.data) - 1) / (math.exp(2 * self.data) + 1)
        output = Value(data=tan, _children=(self,))

        def _backward():
            self.grad += (1 - tan ** 2) * output.grad

        output._backward = _backward

        return output

    def backward(self) -> None:
        topological_order = []
        visited = set()

        topological_order = build_topological_order(
            v=self,
            topological_order=topological_order,
            visited=visited
        )

        # Go one variable at a time and apply the chain rule to get its
        # gradient.
        self.grad = 1.0  # Base case.

        for v in reversed(topological_order):
            v._backward()
