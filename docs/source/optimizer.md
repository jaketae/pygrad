# Optimizer

PyGrad offers a number of different optimizers. These include

- `pygrad.nn.optim.SGD`
- `pygrad.nn.optim.AdaGrad`
- `pygrad.nn.optim.AdaDelta`
- `pygrad.nn.optim.Adam`

Training a model with optimizers can be done by instantiating an `nn.optim.Optimizer` object with some model's parameters. After calling `backward()` on a loss value, simply call `step()` on the optimizer to update the target model's parameters.

```python
# imports assumed

optimizer = nn.optim.Adam(model.params())

for data, labels in train_loader:
    pred = model(data)
    loss = F.softmax_cross_entropy(pred, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

The optimizer will update the model's weights according to the gradient values of each parameter.