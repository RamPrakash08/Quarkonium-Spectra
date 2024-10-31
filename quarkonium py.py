import numpy as np
import math
import scipy.linalg as la
from scipy.special import genlaguerre

# User input for quark type
quark = int(input("Enter the quark type (1 for c, 0 for b): "))
if quark:
    alpha_s = 0.5461  # Strong coupling constant for charmonium
    sigma = 0.1425    # String tension in GeV^2
    V_c = -0.014      # Potential constant in GeV
    m_c = 1.47941     # Charm quark mass in GeV
    rho = 1.19
else:
    alpha_s = 0.345   # Strong coupling constant for bottomonium
    sigma = 0.19      # String tension in GeV^2
    V_c = -0.107      # Potential constant in GeV
    m_c = 4.8         # Bottom quark mass in GeV
    rho = 3.54

reduced_mass = 0.5 * m_c  # Reduced mass in GeV

def radial_wavefunction(n, l, r):
    """Computes radial wavefunction for specified n and l."""
    # if n <= l:
    #     return np.zeros_like(r)
    n = n + l
    a0 = 1 / (alpha_s * m_c)  # Effective Bohr radius
    rho = 2 * r / (n * a0)
    normalization = np.sqrt((2 / (n * a0))**3 * math.factorial(n - l - 1) / (2 * n * math.factorial(n + l)))
    L_n_minus_1 = genlaguerre(n - l - 1, 2 * l + 1)(rho)
    return normalization * np.exp(-rho / 2) * rho**l * L_n_minus_1

def hamiltonian_matrix_element(n1, l, n2, r, dr, s, j):
    """Compute the Hamiltonian matrix element H_{n1, n2} for a given l, s, and j."""
    R_n1 = radial_wavefunction(n1, l, r)
    R_n2 = radial_wavefunction(n2, l, r)
    Ls = 0.5 * (j * (j + 1) - l * (l + 1) - s * (s + 1))
    
    # Potential energy term: <R_n1 | V(r) | R_n2>
    V = np.sum(
        R_n1 * R_n2 * (
            -((4 / 3) * alpha_s / r) + (sigma * r)
            + 32 * math.pi * alpha_s * (0.5 * s * (s + 1) - 0.75) * ((rho / np.sqrt(math.pi))**3) * np.exp(-((rho * r)**2)) / (9 * m_c**2)
            + ((4 * alpha_s * r - sigma) / (2 * m_c**2 * r**2)) * (Ls)
            + (alpha_s / (3 * m_c**2 * r**3)) * (4 / ((2 * l + 3) * (2 * l - 1))) * (s * l * (s + 1) * (l + 1) - 1.5 * Ls - 3 * Ls**2)
        ) * r**2
    ) * dr
    
    # Kinetic energy term with angular component l(l+1)/r^2
    dR_n2 = np.gradient(R_n2, dr)
    d2R_n2 = np.gradient(r**2 * dR_n2, dr) / r**2
    T = -0.5 * np.sum(R_n1 * d2R_n2 * r**2) * dr / reduced_mass
    T += 0.5 * np.sum(R_n1 * R_n2 * l * (l + 1) / r**2 * r**2) * dr / reduced_mass
    
    return T + V

def construct_hamiltonian_matrix(n_basis, l, r, dr, s, j):
    """Constructs the Hamiltonian matrix for a specific l, s, and j combination."""
    H = np.zeros((n_basis, n_basis))
    for n1 in range(1, n_basis + 1):
        for n2 in range(1, n_basis + 1):
            H[n1 - 1, n2 - 1] = hamiltonian_matrix_element(n1, l, n2, r, dr, s, j)
    return H

def find_minimum_eigenvalue_per_combination(n_basis, max_l):
    """Finds the minimum eigenvalue for each (l, s, j) combination."""
    final_r = 500
    r = np.linspace(0.001, final_r, int(final_r * 100))
    dr = r[1] - r[0]
    
    min_eigenvalues = []
    for l in range(0, max_l + 1):
        for s in [0, 1]:  # Spin values
            for j in range(l - s, l + s + 1):  # j goes from l-s to l+s
                if j < 0:
                    continue  # Skip invalid j values
                H = construct_hamiltonian_matrix(n_basis, l, r, dr, s, j)
                eigenvalues, _ = la.eigh(H)
                eigenvalues = np.sort(eigenvalues)
                print(eigenvalues + 2 * m_c + V_c, l, s, j)
                min_eigenvalue = np.min(eigenvalues)
                min_eigenvalue_mev = (2 * m_c + min_eigenvalue + V_c) * 1000  # Convert to MeV
                
                # Store each (l, s, j) minimum eigenvalue with configuration
                min_eigenvalues.append((l, s, j, min_eigenvalue_mev))
    
    return min_eigenvalues

# User input for basis functions and maximum L
n_basis = int(input("Enter the number of basis functions: "))
L = int(input("Enter the maximum value of L: "))

# Find and print minimum eigenvalue for each (l, s, j)
min_eigenvalues = find_minimum_eigenvalue_per_combination(n_basis, L)
print("\nMinimum eigenvalues for each (l, s, j) combination in MeV:")
for l, s, j, eigenvalue in min_eigenvalues:
    print(f"l = {l}, s = {s}, j = {j} => Minimum eigenvalue: {eigenvalue} MeV")

# Find overall minimum
overall_min = min(min_eigenvalues, key=lambda x: x[3])
print(f"\nOverall minimum eigenvalue across all (l, s, j): {overall_min[3]} MeV at (l = {overall_min[0]}, s = {overall_min[1]}, j = {overall_min[2]})")
