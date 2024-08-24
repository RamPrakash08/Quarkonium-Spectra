import numpy as np
import scipy.linalg as la
from scipy.special import genlaguerre

# Constants
HARTREE_TO_EV = 27.2114  # Conversion factor from Hartree to eV
a0 = 1.0  # Bohr radius in atomic units

def hydrogen_radial_wavefunction(n, r):
    """
    Radial wavefunction for the hydrogen atom.
    Uses the radial part of the hydrogen atom wavefunctions.
    """
    Z = 1  # Atomic number for hydrogen
    rho = 2 * Z * r / (n * a0)
    normalization = np.sqrt((2*Z/(n*a0))**3 * np.math.factorial(n-1) / (2*n*np.math.factorial(n)))
    L_n_minus_1 = genlaguerre(n-1, 1)(rho)
    return normalization * np.exp(-rho / 2) * L_n_minus_1

def hamiltonian_matrix_element(n1, n2, r, dr):
    """
    Compute the Hamiltonian matrix element H_{n1, n2}.
    """
    R_n1 = hydrogen_radial_wavefunction(n1, r)
    R_n2 = hydrogen_radial_wavefunction(n2, r)
    
    # Kinetic energy term: <R_n1 | -1/2 * (1/r^2 * d/dr(r^2 d/dr)) | R_n2>
    dR_n2 = np.gradient(R_n2, dr)
    d2R_n2 = np.gradient(r**2 * dR_n2, dr) / r**2
    T = -0.5 * np.sum(R_n1 * d2R_n2 * r**2) * dr
    
    # Potential energy term: <R_n1 | -1/r | R_n2>
    V = -np.sum(R_n1 * R_n2 * r) * dr
    
    return T + V

def construct_hamiltonian_matrix(n_basis, r, dr):
    """
    Construct the Hamiltonian matrix for a given number of basis functions.
    """
    H = np.zeros((n_basis, n_basis))
    for n1 in range(1, n_basis + 1):
        for n2 in range(1, n_basis + 1):
            H[n1-1, n2-1] = hamiltonian_matrix_element(n1, n2, r, dr)
    return H

def find_minimum_eigenvalue(n_basis):
    """
    Find the minimum eigenvalue of the Hamiltonian matrix and convert it to eV.
    """
    # Define the range for r and compute the step size dr
    r = np.linspace(0.01, 20, 2000)  # Increased number of points for better accuracy
    dr = r[1] - r[0]
    
    H = construct_hamiltonian_matrix(n_basis, r, dr)
    eigenvalues, _ = la.eigh(H)
    
    min_eigenvalue_hartree = np.min(eigenvalues)
    min_eigenvalue_ev = min_eigenvalue_hartree * HARTREE_TO_EV
    
    return min_eigenvalue_ev

# Example usage
n_basis = int(input("Enter the number of basis functions: "))
min_eigenvalue = find_minimum_eigenvalue(n_basis)
print(f"The minimum eigenvalue for n_basis = {n_basis} is: {min_eigenvalue} eV")
