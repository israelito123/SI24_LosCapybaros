tienes un merge conflict abierto en el ejercicio de q_learning.

Aquí, la ecuación está bien implementada, el único comentario es que el nombre de la variable no es representativo de la realidad ya que `np.max(self.Q[next_state])` no es la mejor acción siguiente, sino el mejor value function siguiente. La mejor acción siguiente sería `np.argmax(self.Q[next_state])`. Esto no afecta en nada al funcionamiento, solo hago la aclaración.
```
        best_next_action = np.max(self.Q[next_state])
        # TODO: Implementa la actualización de Q-learning usando la ecuación vista en clase
        self.Q[state][action] = self.Q[state][action] +self.alpha*(reward + self.gamma*best_next_action-self.Q[state][action])

```