# Quantum Trajectories in Pilot Wave Theory
###### Under the supervision of [Professor Ivan Ivanov](https://euclid.vaniercollege.qc.ca/~iti/) at [Vanier College](https://www.vaniercollege.qc.ca/).

![alt text](https://github.com/IsolatedSingularity/Quantum-Trajectories/blob/main/Plots/RectangularPotentialWell.png)

## Objective

Pilot wave theory, also known as Bohmian mechanics, emerges as a compelling reformulation of quantum mechanics by providing deterministic trajectories for quantum particles, challenging the probabilistic nature of standard quantum theory. This interpretation unveils the wave-particle duality in a tangible way, aligning with classical intuitions of particles having definite positions and velocities, while preserving notions of causality and determinism often missing in quantum physics. One interesting result of this theory is being able to generate all possible quantum trajectories of the particle between initial and final states, trajectories which are only implicitly computed in the path integral formulation of quantum mechanics. 

We look at computing quantum trajectories for particles interacting with certain potentials, either reflecting off or tunneling through depending on the amplitude of the potential. These computations, however, have a high time complexity due to the combinatorial nature of the class of trajectories the particle takes, and so it would be useful to have algorithms which compute them efficiently for arbitrary potentials. We address this challenge through the implementation of recurrent neural networks (RNNs) that is trained by computing quantum trajectories with the Crank-Nicolson integrator method for solving the modified Schrödinger PDE equation. This results in an efficient script that computes the network of trajectories the quantum particle can evolve to after interacting with an arbitrary potential.

## Code Functionality

*Algorithm.py* simulates the behavior of a quantum particle in a potential well using a combination of numerical methods and recurrent neural networks. First, it defines simulation parameters, such as the particle's mass, time-related parameters, and the spatial domain. The potential well is established as an upright well with specific characteristics. Next, it initializes the initial wave function of the particle as a Gaussian distribution, which represents the particle's probability distribution across space. The code then sets up matrices and equations to implement absorbing boundary conditions using Crank-Nicolson scheme. These conditions ensure that the wave function remains stable at the boundaries of the simulation. The RNN is introduced into the simulation. It uses TensorFlow to create a neural network with a SimpleRNN layer. This RNN aims to predict the future values of Psi based on its current values, effectively modeling the quantum particle's behavior over time. The RNN model is trained using the Psi data generated by the simulation. After training, it can predict Psi values for subsequent time steps. Finally, the code computes the probability density of the particle at each time step using the predicted Psi values. It also calculates the action function and velocity field, allowing for further analysis of the particle's behavior.

The report covering all the theory and code can be found in the main repository as a PDF file.

## Caveats

The absorbing boundary conditions might not be perfect, potentially causing boundary effects that affect the accuracy of the simulation's outcomes. Moreover, the spatial and temporal discretization steps (dx and dt) can introduce errors, especially when dealing with fine details of the quantum system.

## Next Steps

At this stage, we've clarified the problem statement and established the fundamental code framework. Potential additions to the code would include incorporatating quantum effects like entanglement into the simulation for a deeper understanding of complex quantum systems. Moreover one could optimize the code for parallel processing to handle larger and more computationally intensive simulations efficiently.
