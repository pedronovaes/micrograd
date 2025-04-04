"""
Mathematical expressions classes to build a multilayer perceptron (MLP)
neural.
"""

from random import uniform
from typing import List
from src.tensor import Value


class Neuron:
    """
    Emulates a neuron with specific number of weights and one bias that control
    the overall trigger happiness. Both weights and bias are initialized with
    uniform dist-based random numbers between -1 and 1.

    Attributes:
        nin: number of inputs of the neuron (how many inputs come to a neuron).
    """

    def __init__(self, nin: int) -> None:
        self.W = [Value(data=uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(data=uniform(-1, 1))

    def __call__(self, x):
        """
        Neuron forward pass by applying a pairwise multiplication: W * x + b.
        """

        act = sum(wi * xi for wi, xi in zip(self.W, x)) + self.b

        # Non-linear function.
        output = act.tanh()

        return output

    def parameters(self) -> List[Value]:
        """Returns a list of all neuron params (weights and bias)."""

        return self.W + [self.b]

    def __repr__(self) -> str:
        return f"Neuron({len(self.W)})"


class Layer:
    """
    A layer is just a list of neurons. This class also implements the forward
    pass for all the neurons in a layer.

    Attributes:
        nin: number of inputs of each neuron in the layer.
        nout: number of neurons in the layer.
    """

    def __init__(self, nin: int, nout: int) -> None:
        self.neurons = [Neuron(nin=nin) for _ in range(nout)]

    def __call__(self, x) -> List[Value] | Value:
        """Forward pass to evaluate each neuron independently."""

        outs = [n(x=x) for n in self.neurons]

        return outs[0] if len(outs) == 1 else outs

    def parameters(self) -> List[Value]:
        """Returns a list of all neuron params in the layer."""

        return [
            parameter
            for neuron in self.neurons
            for parameter in neuron.parameters()
        ]

    def __repr__(self) -> str:
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP:
    """
    Build a multilayer perceptron neural network, that consists of a fully
    connected neurons with nonlinear activation functions organized in layers.

    Attributes:
        nin: number of inputs of the neural net.
        nouts: list containing how many layers and how many neurons has in
            each layer.
    """

    def __init__(self, nin: int, nouts: List[int]) -> None:
        size = [nin] + nouts
        self.layers = [
            Layer(nin=size[i], nout=size[i + 1])
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        """Full forward pass."""

        for layer in self.layers:
            x = layer(x=x)

        return x

    def parameters(self) -> List[Value]:
        """Returns a list of all mlp parameters."""

        return [
            parameter
            for layer in self.layers
            for parameter in layer.parameters()
        ]

    def zero_grad(self) -> None:
        """Set to zero the gradients of all the mlp parameters."""

        for parameter in self.parameters():
            parameter.grad = 0.0

    def __repr__(self) -> str:
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
