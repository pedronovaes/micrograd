# micrograd

A tiny **autograd** (automatic gradient) engine. This approach leverages the fact that even biggest, most complicated program must be built from a small set of primitive operations such as addition, multiplication, or trigonometric functions. The **chain rule** allows us to take full advantage of this property.

This repo is inspired by Andrej Karpathy's course and implements the backpropagation algorithm (more specifically the *reverse-mode* autodiff) over a dynamically built DAG (directed acyclic graph). Also, it implements a simple pytorch-like neural networks (a multi-layer perceptron).
