# Info
This repository is for keeping track of "learning machine learning".

# Notes
### Gradient Descent
Gradient descent is based around the concept of figuring out the cost
function (error) of predictions compared to their actual values. The
derivative or slope of the cost function is used to gradually move the
predictions closer to the local minima. A learning rate scalar is used
to avoid overshooting, while keeping gradient descent convergence as
fast and efficient as possible.

### NumPy
NumPy optimizes vector math by allowing normal operations to be done on
entire arrays in one step without using a python for loop. Given the
NumPy array `xList`, you can use an equation like
```
prediction = m * xList + b
```
to generate all the predicted values for linear regression based on your current weights.


# Todo List / Goals
- Finish 3Blue1Brown playlists for linear algebra, calculus,
and machine learning
- Transition from linear regression to:
  - Multivariable linear regression✅
  - Nonlinear regressions
  - Deep reinforcement learning
  - Text generation