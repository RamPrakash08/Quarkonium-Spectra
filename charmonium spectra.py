
import numpy as np
import scipy.linalg as la
import scipy.special as sp
from scipy.special import hermite, genlaguerre
import scipy.integrate as integrate
import matplotlib.pyplot as plt

alpha_s = 0.5461 # Strong coupling constant (dimensionless, same in any unit system)
sigma = 0.1425 #GeV^2  # String tension in GeV^2
 
m_c = 1.47941 # charm quark mass in GeV
reduced_mass = 0.5 * m_c # Reduced mass of the system in GeV
omega_by_c = 0.24

# Define the radial wavefunction for the bottomonium system
def bottomonium_radial_wavefunction(n,l,r):
    #rho = m * omega / hbar * r ** 2
    # we choose m * omega / hbar = (sigma * fm**2)/alpha_s
    
    rho = reduced_mass*omega_by_c* (r**2)
    laguerre_polynomial = sp.genlaguerre(n - 1, l + 0.5)(rho)
    R_nl = r**l * np.exp(-rho / 2) * laguerre_polynomial
    integrand = (R_nl * r)**2
    integral = integrate.simps(integrand, r)
    norm_factor = np.sqrt(integral)
    return R_nl / norm_factor

def hamiltonian_matrix_element(n1, l, n2, r, dr):
    """
    Compute the Hamiltonian matrix element H_{n1, n2}.
    """
    R_n1 = bottomonium_radial_wavefunction(n1, l, r)
    R_n2 = bottomonium_radial_wavefunction(n2, l, r)
    
    # Potential energy term: <R_n1 | V(r) | R_n2>
    V = -(np.sum(R_n1 * R_n2 * (((4 / 3) * alpha_s / (r)) - (sigma * (r)) + 0.75 * r*r * 32*np.pi*alpha_s/(9*m_c**2) \
            *(sigma/np.sqrt(np.pi))**3 * np.exp(-sigma**2 * r*r)) * r**2) * dr)
    
    T = -1* (np.sum(R_n1 * R_n2 * (0.5*reduced_mass*(omega_by_c**2)*(r**2)) * r**2 * dr))
    
    return T+V

def construct_hamiltonian_matrix(n_basis, l, r, dr):
    """
    Construct the Hamiltonian matrix for a given number of basis functions.
    """
    H = np.zeros((n_basis, n_basis))
    for n1 in range(1, n_basis + 1):
        for n2 in range(1, n_basis + 1):
            H[n1-1, n2-1] = hamiltonian_matrix_element(n1, l, n2, r, dr)
            if n1 == n2:
                H[n1-1, n2-1] += (2*(n2 -1) + 1.5 + l)*omega_by_c
    return H

def find_minimum_eigenvalue(n_basis, l):
    """
    Find the minimum eigenvalue of the Hamiltonian matrix and convert it to MeV.
    """
    # Define the range for r and compute the step size dr
    final_r = 500
    r = np.linspace(0.001, final_r, int(final_r * 100)) # Increased number of points for better accuracy
    dr = (r[1] - r[0])
    
    H = construct_hamiltonian_matrix(n_basis, l, r, dr)
    #print(H)
    eigenvalues, _ = la.eigh(H)
    print((eigenvalues + 2*m_c))
    min_eigenvalue_hartree = np.min(eigenvalues)
    min_eigenvalue_ev = min_eigenvalue_hartree
    
    return min_eigenvalue_ev

# Example usage
n_basis = int(input("Enter the number of basis functions: "))
l = int(input("Enter the value of l: "))
min_eigenvalue = find_minimum_eigenvalue(n_basis, l)
print(f"The minimum eigenvalue for n_basis = {n_basis}, l = {l} is: {(2 * m_c + min_eigenvalue)*1000} MeV")
print(f"The minimum eigenvalue for n_basis = {n_basis}, l = {l} is: {(min_eigenvalue)} GeV")

def potential(r):
    return -(((4 / 3) * alpha_s / (r)) - (sigma * (r)))

# Plot the potential
r_values = np.linspace(0.5, 5, 1000) # r from 0.01 to 3 fermi
y = np.zeros_like(r_values)
V_values = 1 * potential(r_values)
plt.plot(r_values, V_values, label='Potential V(r)', color='black')
plt.plot(r_values, y, color='black')

# Plot all basis functions
for n in range(1, n_basis + 1):
    R_values = bottomonium_radial_wavefunction(n, l, r_values)
    plt.plot(r_values, R_values, label=f'R_{n}(r)')

plt.xlabel('r (fm)')
plt.ylabel('R_nl(r)')
plt.title('Radial Wavefunctions and Potential')
plt.legend()
plt.show()