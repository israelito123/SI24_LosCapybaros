Aquí, podrían usar la función anteriomente definida (mse_loss) para sacar el costo en lugar de estarlo recalculando cada vez. Cuando se use numpy, eviten usar for loops y aprovechense del broadcasting.

```
for index in range(100):
    for subindex in range(100):
        w = np.array([W[0, index], W[1, subindex]])
        y_pred = np.dot(x_train, w)
        calc_cost[index, subindex] = np.mean(np.square(y_train-y_pred))
```

para que les saliera la gráfica como era tendrían que cambiar la línea donde asignan w a:

```
for index in range(100):
    for subindex in range(100):
        # w = np.array([W[0, index], W[1, subindex]])
        w = np.array([w0[index], w1[subindex]])
        y_pred = np.dot(x_train, w)
        calc_cost[index, subindex] = np.mean(np.square(y_train-y_pred))
```

porque W es como una listota de todas las posibles combinaciones. Es decir W[:, 0] es la primera linea, W[:, 1] es la segunda y así sucesivamente. Entonces no pueden indexarlo con index y subindex. Pero w0 y w1 son 100 números de entre -10 a 10, entonces ese si se puede indexar como lo hicieron.
