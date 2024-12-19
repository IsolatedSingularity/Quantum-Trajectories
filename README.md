# Quantum Trajectories in Pilot Wave Theory
###### Under the supervision of [Professor Ivan Ivanov](https://euclid.vaniercollege.qc.ca/~iti/) at [Vanier College](https://www.vaniercollege.qc.ca/), and in collaboration with [Nicolas Desjardins-Proulx](https://kildealab.com/author/nicolas-desjardins-proulx/).

![Rectangular Potential Well](https://github.com/IsolatedSingularity/Quantum-Trajectories/blob/main/Plots/RectangularPotentialWell.png?raw=true)

## Objective

Pilot wave theory, also known as Bohmian mechanics, provides a deterministic reformulation of quantum mechanics, proposing that particles follow defined trajectories influenced by a guiding wave. This interpretation contrasts with the probabilistic nature of standard quantum mechanics, aligning better with classical notions of causality and determinism. Importantly, pilot wave theory allows computation of all possible quantum trajectories for particles between initial and final states, which are only implicitly handled in path integral quantum mechanics.

This project focuses on simulating quantum trajectories for particles interacting with potential barriers, either reflecting or tunneling through based on the barrier’s amplitude. Due to the combinatorial nature of trajectory computation, these simulations are computationally expensive. To address this, we implemented recurrent neural networks (RNNs) trained on data generated using the Crank-Nicolson method for solving the modified Schrödinger equation, enabling efficient computation of trajectories for arbitrary potentials.

---

## Theoretical Background

Pilot wave theory introduces deterministic trajectories governed by the guiding equation:

$$
\mathbf{v} = \frac{\nabla S}{m},
$$

where:

- \(\mathbf{v}\): Velocity field of the particle.
- \(S\): Action function obtained from the wavefunction \(\Psi = R e^{iS/\hbar}\).

The Schrödinger equation describes the evolution of \(\Psi\):

$$
i \hbar \frac{\partial \Psi}{\partial t} = -\frac{\hbar^2}{2m} \nabla^2 \Psi + V \Psi,$$

where \(V(x)\) is the potential energy function. Bohmian mechanics extends this by extracting trajectories from the velocity field:

$$
x(t + \Delta t) = x(t) + v(x(t)) \Delta t.$$

In tunneling scenarios, the particle’s trajectory depends on the energy \(E\) relative to the potential \(V(x)\):

- If \(E > V(x)\): The particle reflects.
- If \(E < V(x)\): The particle tunnels through.

The Crank-Nicolson method numerically solves the Schrödinger equation to compute the wavefunction \(\Psi\) at discrete time steps, while the RNN predicts future \(\Psi\) states for improved efficiency.

---

## Code Functionality

The code (*Algorithm.py*) models quantum particles interacting with potential wells using both numerical methods and RNNs. Key components include:

1. **Simulation Parameters:**
   - Defines time and spatial domains, particle mass, and boundary conditions.

```python
# Simulation parameters
final_time = 5
mesh_size = 3001
xi, xf = -20, 20
Xgrid = np.linspace(xi, xf, mesh_size)
dx = (xf - xi) / (mesh_size - 1)
dt = final_time / 5000  # Time step
```

2. **Potential Function:**
   - Sets up the potential well where particles interact.

```python
# Potential function
def V(x):
    return 8 if 2 <= x <= 6 else 0

# Create potential grid
potential = np.array([V(x) for x in Xgrid])
```

3. **Initial Wavefunction:**
   - Initializes \(\Psi\) as a Gaussian distribution centered at \(\mu_0\).

```python
# Initial wavefunction
k0, sigma0, mu0 = 3, 1, -9
Psi = (1 / (2 * np.pi * sigma0**2)) * np.exp(1j * k0 * Xgrid - ((Xgrid - mu0) / (2 * sigma0))**2)
```

4. **Absorbing Boundary Conditions:**
   - Implements Crank-Nicolson boundary conditions to stabilize \(\Psi\) near simulation edges.

```python
# Absorbing boundary conditions
alpha = (1j * dt) / (2 * dx**2)
U1 = sp.diags([-alpha, 1 + 2*alpha, -alpha], [-1, 0, 1], shape=(mesh_size, mesh_size)).tocsc()
```

5. **Crank-Nicolson Integration:**
   - Evolves \(\Psi\) over time using a sparse matrix solver.

```python
# Time evolution of Psi
for t in range(time_steps - 1):
    Psi[t + 1, :] = linalg.spsolve(U1, U2.dot(Psi[t, :]))
```

6. **Recurrent Neural Network (RNN):**
   - Trains an RNN to predict future \(\Psi\) states, reducing computational overhead.

```python
# Define RNN
model = Sequential([
    SimpleRNN(64, return_sequences=True, input_shape=(mesh_size, 1)),
    Dense(mesh_size, activation='linear')
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(Psi[:-1], Psi[1:], epochs=10, verbose=1)
```

7. **Trajectory Computation:**
   - Calculates particle trajectories from the velocity field.

```python
# Compute velocity field
v = np.gradient(np.unwrap(np.angle(Psi)), axis=1) / (mass * dx)

# Compute trajectories
for l in range(nt):
    for t in range(1, time_steps):
        x[l, t] = x[l, t-1] + v[t-1, int((x[l, t-1] - xi) / dx)] * dt
```

---

## Caveats

- **Boundary Conditions:** Absorbing boundaries may introduce artifacts, affecting trajectory accuracy.
- **Discretization Errors:** Finer spatial and temporal grids improve accuracy but increase computation time.
- **Neural Network Training:** Model performance depends on training data quality and hyperparameter tuning.

---

## Next Steps

- [x] Extend trajectory computation to multi-dimensional potentials.
- [ ] Incorporate quantum effects like entanglement into trajectory predictions.
- [ ] Optimize code for parallel processing to handle larger simulations.
- [ ] Validate neural network predictions against analytical solutions.

---

> [!TIP]
> Regularly validate RNN predictions with Crank-Nicolson solutions to ensure accuracy.

> [!NOTE]
> Detailed derivations of trajectory equations and RNN architectures are available in the repository PDF.

