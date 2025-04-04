# micrograd

A tiny **autograd** (automatic gradient) engine. This approach leverages the fact that even biggest, most complicated program must be built from a small set of primitive operations such as addition, multiplication, or trigonometric functions. The **chain rule** allows us to take full advantage of this property.

This repo is inspired by Andrej Karpathy's course and implements the backpropagation algorithm (more specifically the *reverse-mode* autodiff) over a dynamically built DAG (directed acyclic graph). Also, it implements a simple pytorch-like neural networks (a multi-layer perceptron).

## Usage example to train a mlp neural net.

You are able to train a multilayer perceptron neural net using this repo. Just imports some modules, defines a dataset:

```python
from micrograd.src.nn import MLP
from micrograd.src.utils import mean_squared_error

X = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]

y = [1.0, -1.0, -1.0, 1.0]
```

Create a simple 2-layer MLP neural net. Each layer contains 4 neurons:

```python
model = MLP(nin=3, nouts=[4, 4, 1])
```

Define a training loop and voilà:

```python
for iterations in range(100):

    # Forward pass.
    y_pred = [model(x) for x in X]

    # Measure loss.
    loss = mean_squared_error(y_true=y, y_pred=y_pred)

    # Backward pass.
    model.zero_grad()
    loss.backward()

    # Update.
    for p in model.parameters():
        p.data -= 0.01 * p.grad
```

After training, you can see the model convergence:

```python
print(y_pred)
```

```
[Value(data=0.9643692809956175),
 Value(data=-0.9508870212284644),
 Value(data=-0.8449294005887718),
 Value(data=0.897195035932002)]
```

You can find a full demo of training a classifier in the notebook `demo.ipynb` (just copied from [Karpathy repo](https://github.com/karpathy/micrograd/)).
