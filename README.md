# Quantum Trajectories Pilot Wave Theory
###### Under the supervision of [Professor Ivan Ivanov](https://euclid.vaniercollege.qc.ca/~iti/) at [Vanier College](https://www.vaniercollege.qc.ca/).

![alt text](https://github.com/IsolatedSingularity/Quantum-Trajectories/blob/main/Plots/RectangularPotentialWell.png)

## Objective

Phase transitions in the very early universe cause spontaneous symmetry breaking events which create topological defects. Linear topological defects associated with the U(1) symmetry group are known as **cosmic strings** and create high energy signals in the form of wakes as they propagate through spacetime. Their presence thus is of interest to study the high energy structure of the universe and the standard model.

The strings occur in a class of renormalizable quantum field theories and are stable under specific conditions on the spacetime manifold's homotopy groups. Their dynamics are given by the Nambu-Gotto action, much like bosonic excitations in string theory. What's more is that gravitational backreaction during the early universe causes primordial ΛCDM noise which hides the string signal. Thus the purpose of this repository is to develop statistics to efficiently extract the cosmic string signal admist the non-linear noise through the framework of 21cm cosmology.

## Code Functionality

*Algorithm.py* simulates the behavior of a quantum particle in a potential well using a combination of numerical methods and recurrent neural networks (RNNs). First, it defines simulation parameters, such as the particle's mass, time-related parameters, and the spatial domain. The potential well is established as an upright well with specific characteristics. Next, it initializes the initial wave function (Psi) of the particle as a Gaussian distribution, which represents the particle's probability distribution across space. The code then sets up matrices and equations to implement absorbing boundary conditions (ABC) using Crank-Nicolson scheme. These conditions ensure that the wave function remains stable at the boundaries of the simulation. The RNN is introduced into the simulation. It uses TensorFlow to create a neural network with a SimpleRNN layer. This RNN aims to predict the future values of Psi based on its current values, effectively modeling the quantum particle's behavior over time. The RNN model is trained using the Psi data generated by the simulation. After training, it can predict Psi values for subsequent time steps. Finally, the code computes the probability density of the particle at each time step using the predicted Psi values. It also calculates the action function and velocity field, allowing for further analysis of the particle's behavior.

The report covering all the theory and code can be found in the main repository as a PDF file.

## Caveats

The absorbing boundary conditions might not be perfect, potentially causing boundary effects that affect the accuracy of the simulation's outcomes. Moreover, the spatial and temporal discretization steps (dx and dt) can introduce errors, especially when dealing with fine details of the quantum system.

## Next Steps

At this stage, we've clarified the problem statement and established the fundamental code framework. Potential additions to the code would include incorporatating quantum effects like entanglement into the simulation for a deeper understanding of complex quantum systems. Moreover one could optimize the code for parallel processing to handle larger and more computationally intensive simulations efficiently.
