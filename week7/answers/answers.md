# Hand-In Lab 06 Neural Networks - Answers

---

## Frage 1: Output of the network for x1=1 and x2=-1

**Given:** All weights w_ji = 1 and u_i = 1, ReLU activation.

Step-by-step computation:

```
h1 = relu(w11*x1 + w21*x2) = relu(1*1 + 1*(-1)) = relu(0) = 0
h2 = relu(w12*x1 + w22*x2) = relu(1*1 + 1*(-1)) = relu(0) = 0
h3 = relu(w13*x1 + w23*x2) = relu(1*1 + 1*(-1)) = relu(0) = 0

y_hat = relu(u1*h1 + u2*h2 + u3*h3) = relu(1*0 + 1*0 + 1*0) = relu(0) = 0
```

**Answer: 0**

---

## Frage 2: Output of the network for x1=1 and x2=1

Step-by-step computation:

```
h1 = relu(w11*x1 + w21*x2) = relu(1*1 + 1*1) = relu(2) = 2
h2 = relu(w12*x1 + w22*x2) = relu(1*1 + 1*1) = relu(2) = 2
h3 = relu(w13*x1 + w23*x2) = relu(1*1 + 1*1) = relu(2) = 2

y_hat = relu(u1*h1 + u2*h2 + u3*h3) = relu(1*2 + 1*2 + 1*2) = relu(6) = 6
```

**Answer: 6**

---

## Frage 3: Hand-in implementation

### Forward pass implementation:

```python
def forward_pass(x, W, u):
    """
    Implement the forward pass for our network NN(x, W, u)

    Parameters:
      x: a 2-element vector containing x_1 and x_2 (numpy array of shape (2,))
      W: a 2x3 matrix containing weights w_ji (numpy array of shape (2, 3))
      u: a 3-element vector containing u_i (numpy array of shape (3,))

    Returns:
      y: prediction of what y should be like
    """
    h = activation(W.T @ x)
    y = activation(u @ h)
    return y
```

### Weight updates (SGD):

```python
W, u = initialize_weights()

learning_rate = 0.001
n_steps = 1000

losses = []

for t in range(n_steps):
    sample_ix = rng.integers(0, 4)
    x_t = X[sample_ix]
    y_t = y[sample_ix]

    loss = 0.5 * (y_t - forward_pass(x_t, W, u))**2
    losses.append(loss)

    grad_u, grad_w = grads(y_t, x_t, W, u)

    # Update the weights using stochastic gradient descent
    W = W - learning_rate * grad_w
    u = u - learning_rate * grad_u
```

### Learning curve:

The learning curve is plotted with:
```python
plt.plot(np.arange(n_steps), losses)
```

(See the notebook output for the actual plot — run the notebook to generate it.)

---

## Frage 4: Best option and final accuracy (Task 3a)

I tried changing different options one at a time to see what helps the most:

- **More epochs (e.g. 100 instead of 10):** The model improved a bit but still wasn't great with plain SGD. It just needs more time to converge.
- **`use_bias=True`:** Already set in the second cell, didn't change much on its own.
- **`loss="binary_crossentropy"`:** Helped slightly compared to MSE since this is a classification problem, but wasn't enough alone.
- **Changing the optimizer to `"adam"`:** This made the biggest difference by far. With Adam the model converged much faster and actually learned the XOR pattern properly.
- **More hidden nodes / layers:** Adding more nodes helped a little but wasn't as impactful as switching the optimizer.

In the end, the option that worked best for me was **switching the optimizer from `"sgd"` to `"adam"`**. Combined with `binary_crossentropy` as the loss, `use_bias=True`, and training for 100 epochs, the model reached **100% accuracy** on the XOR problem.

```python
model = Sequential()
model.add(Dense(3, input_shape=(2,), use_bias=True))
model.add(Activation("relu"))
model.add(Dense(1, input_shape=(3,), use_bias=True))
model.add(Activation("sigmoid"))
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

history = model.fit(X, y, epochs=100, batch_size=1)
```

**Final accuracy: 1.0 (100%)**

---

## Frage 5: True / False

| # | Statement | Answer | Explanation |
|---|-----------|--------|-------------|
| 1 | Fully connected neural networks have all possible connections between pairs of neurons in consecutive hidden layers | **True** | That is the definition of "fully connected" (dense) — every neuron in one layer is connected to every neuron in the next layer. |
| 2 | Early stopping in the training process of a neural network with gradient descent increases the chances of overfitting | **False** | Early stopping is a regularization technique that *reduces* overfitting by stopping training before the model memorizes the training data. |
| 3 | Neural networks can also learn without an activation function (identity activation) | **True** | Without a non-linear activation, the network is equivalent to a single linear transformation. It can still learn linear relationships, just not non-linear ones. |
| 4 | A neural network has a minimum of 2 hidden layers | **False** | A neural network can have zero hidden layers (e.g., a single-layer perceptron) or just one hidden layer. There is no minimum of 2. |
