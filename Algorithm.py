# Listing 1: Simulation Parameters
import numpy as np

# Planck's constant is set to 1
mass = 0.5  # Mass of the particle

# Time-related parameters
final_time = 5
time_steps = 5000
dt = final_time / time_steps

# Space/mesh-related parameters
xi = -20  # Left endpoint
xf = 20   # Right endpoint
mesh_size = 3001
dx = (xf - xi) / (mesh_size - 1)

# Listing 2: Setting up the potential
def V(x):
    return 8 if 2 <= x <= 6 else 0

# Store the potential in a grid
potential = np.array([V(xi + i * dx) for i in range(mesh_size)])

# Define the X grid and the Psi array
Xgrid = np.linspace(xi, xf, mesh_size)
Psi = np.zeros((time_steps, mesh_size), complex)

# Parameters for the Initial Distribution
k0 = 3          # wave number
sigma0 = 1      # standard deviation of the initial Gaussian
mu0 = -9        # where the initial Gaussian is centered

Psi[0, :] = (1 / (2 * np.pi * sigma0 ** 2)) * np.exp((1j) * k0 * Xgrid - ((Xgrid - mu0) / (2 * sigma0)) ** 2)

# Listing 3: Setting up the Absorbing Boundary Conditions
q = k0 / mass
h1 = 3 * mass * q
h2 = 3 * (mass * q ** 2) * (q ** 2)
h3 = (mass * q ** 3) * (q ** 3)
ai = h2 / (2 * mass) - V(xi)
af = h2 / (2 * mass) - V(xf)
bi = h3 / (2 * mass) - h1 * V(xi)
bf = h3 / (2 * mass) - h1 * V(xf)

beta1 = -(1j) * ai / (2 * dx) + 1 / (dt * dx) - (1j) * h1 / (2 * dt) - bi / 4
beta2 = (1j) * ai / (2 * dx) - 1 / (dt * dx) - (1j) * h1 / (2 * dt) - bi / 4
beta3 = (1j) * ai / (2 * dx) + 1 / (dt * dx) - (1j) * h1 / (2 * dt) + bi / 4
beta4 = -(1j) * ai / (2 * dx) - 1 / (dt * dx) - (1j) * h1 / (2 * dt) + bi / 4
zeta1 = -(1j) * af / (2 * dx) + 1 / (dt * dx) - (1j) * h1 / (2 * dt) - bf / 4
zeta2 = (1j) * af / (2 * dx) - 1 / (dt * dx) - (1j) * h1 / (2 * dt) - bf / 4
zeta3 = (1j) * af / (2 * dx) + 1 / (dt * dx) - (1j) * h1 / (2 * dt) + bf / 4
zeta4 = -(1j) * af / (2 * dx) - 1 / (dt * dx) - (1j) * h1 / (2 * dt) + bf / 4

# Listing 4: Making the matrix U1 and U2
import scipy.sparse as sp

ones = np.ones((mesh_size), complex)
alpha = (1j) * dt / (2 * dx * dx)

xis = np.array([2 * mass + (1j) * dt / (dx * dx) + (1j) * mass * dt * potential[i] for i in range(mesh_size)])
xis[0] = beta1
xis[mesh_size - 1] = zeta1

up = -alpha * ones
up[1] = beta2
down = -alpha * ones
down[mesh_size - 2] = zeta2

gamma = np.array([2 * mass - (1j) * dt / (dx * dx) - (1j) * mass * dt * potential[i] for i in range(mesh_size)])
gamma[0] = beta3
gamma[mesh_size - 1] = zeta3

ups = alpha * ones
ups[1] = beta4
downs = alpha * ones
downs[mesh_size - 2] = zeta4

diags = np.array([-1, 0, 1])
vecs1 = np.array([down, xis, up])
vecs2 = np.array([downs, gamma, ups])
U1 = sp.diags(vecs1, diags, (mesh_size, mesh_size))
U1 = U1.tocsc()
U2 = sp.diags(vecs2, diags, (mesh_size, mesh_size))

# Listing 5: Solving for Psi
import scipy.linalg as linalg

LU = linalg.splu(U1)
for i in range(time_steps - 1):
    b = U2.dot(Psi[i, :])
    Psi[i + 1, :] = LU.solve(b)

# Compute probability density
Density = np.abs(Psi * np.conj(Psi))

# Listing 7: Utility method to compute derivatives
def derivative(array, dim, dd):
    leng = array.shape[dim]
    der = np.zeros_like(array)
    for i in range(1, leng - 1):
        indx = [Ellipsis] * array.ndim
        indx[dim] = i - 1
        indxr = [Ellipsis] * array.ndim
        indxr[dim] = i + 1
        der[indx] = (array[indxr] - array[indx]) / (2 * dd)
        indx0 = [Ellipsis] * array.ndim
        indx1 = [Ellipsis] * array.ndim
        indx0[dim] = 0
        indx1[dim] = 1
        der[indx0] = (array[indx1] - array[indx0]) / dd
        indxm1 = [Ellipsis] * array.ndim
        indxm2 = [Ellipsis] * array.ndim
        indxm1[dim] = -1
        indxm2[dim] = -2
        der[indxm1] = (array[indxm1] - array[indxm2]) / dd
    return der

# Listing 8: Computing the action function
S = np.angle(Psi)
for i in range(time_steps):
    S[i, :] = np.unwrap(S[i, :])

DisplayActionFunction = False
if DisplayActionFunction:
    figb, axb = plt.subplots()
    plt.axis([xi, xf, -100, 100])
    lineb, = axb.plot(Xgrid, S[0, :])

def animateb(i):
    lineb.set_ydata(S[i, :])  # update the data
    return lineb,

def init():
    lineb.set_ydata(np.ma.array(Xgrid, mask=True))
    return lineb,

# Listing 9: Extracting the velocity field
v = derivative(S, 1, mass * dx)

# Listing 10: Computing the trajectories
nt = 1000  # This is how many trajectories will be computed
x = np.zeros((nt, time_steps))
x[:, 0] = np.linspace(mu0 - 3 * sigma0, mu0 + 3 * sigma0, nt)

for l in range(nt):
    for i in range(1, time_steps):
        loca = (x[l, i - 1] - xi) / dx
        k = int(np.floor(loca))

        if k > mesh_size - 2:
            x[l, i:] = xf
        elif k < 0:
            x[l, i:] = xi
        else:
            x[l, i] = x[l, i - 1] + (
                v[i - 1, k] * (loca - k) + v[i - 1, k + 1] * (1 - loca + k)) * dt

# Transpose X for ease of use
x = np.transpose(x)
