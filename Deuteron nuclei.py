import numpy as np
import matplotlib.pyplot as plt

m_p = 1836.15267343  # Proton mass in atomic units
m_n = 1838.68366173  # Neutron mass in atomic units
mu_D = (m_p * m_n) / (m_p + m_n)  # Reduced mass of the deuteron in atomic units
a0 = 1.0 / mu_D  # Bohr radius in atomic units for deuterium

# Potential parameters
c = 0.4  # Convert c = 0.4 fm to atomic units (Bohr radius)
r_21fm = 2.1 # Convert 2.1 fm to atomic units (Bohr radius)
V_40MeV = -40   # -40 MeV converted to Hartree

def potential_energy(r):
    """Define the potential energy V(r) based on the specified conditions."""
    V = np.zeros_like(r)
    V[(r >= c) & (r <= r_21fm)] = V_40MeV
    return V

def trial_wavefunction(r, alpha, p):
    """Trial wavefunction that smoothly transitions to zero at r = c."""
    return np.where(r >= c, (r - c)**p * np.exp(-alpha * (r - c)), 0)

def energy_functional(alpha, p, r, dr):
    """Calculate the energy functional for given alpha and p."""
    psi = trial_wavefunction(r, alpha, p)
    dpsi_dr = np.gradient(psi, dr)
    d2psi_dr2 = np.gradient(r*r*dpsi_dr, dr)
    
    kinetic = -0.5 * ((197.3**2)/931.5)* (1/mu_D) * np.sum(psi * d2psi_dr2) * dr
    potential = np.sum(potential_energy(r) * (psi**2)*(r**2)) * dr
    
    normalization = np.sum((psi**2)*(r**2)) * dr
    
    energy = (potential + kinetic) / normalization
    return energy

def monte_carlo_minimization(r, dr, num_steps=1000, alpha_init=0.5, p=1):
    """Perform Monte Carlo minimization to find the optimal alpha."""
    alpha = alpha_init
    best_alpha = alpha
    best_energy = energy_functional(alpha, p, r, dr)
    
    for step in range(num_steps):
        new_alpha = alpha + np.random.normal(0, 0.01)
        if new_alpha <= 0:  # alpha must be positive
            continue
        new_energy = energy_functional(new_alpha, p, r, dr)
        if new_energy < best_energy:
            best_energy = new_energy
            best_alpha = new_alpha
    
    return best_alpha, best_energy

# Define the range for r
r = np.linspace(0.01, 20, 2000)  # Range of r from 0.01 to 20 atomic units
dr = r[1] - r[0]

# Perform Monte Carlo minimization
alpha_opt, min_energy = monte_carlo_minimization(r, dr)

# Convert energy to MeV for output
min_energy_mev = min_energy 

print(f"Optimal alpha: {alpha_opt}")
print(f"Minimum energy: {min_energy_mev} MeV")

# Plot the trial wavefunction and final wavefunction
plt.figure(figsize=(10, 6))
plt.plot(r, trial_wavefunction(r, 0.5, 2), label='Trial Wavefunction (initial)')
plt.plot(r, trial_wavefunction(r, alpha_opt, 2), label='Final Wavefunction (after minimization)')
plt.axvline(c, color='r', linestyle='--', label=f"r = {c:.3f} fermi (c)")
plt.axvline(r_21fm, color='g', linestyle='--', label=f"r = {r_21fm:.3f} fermi (2.1 fm)")

plt.title("Trial and Final Wavefunctions")
plt.xlabel("r (fm)")
plt.ylabel("Wavefunction")
plt.legend()
plt.grid(True)
plt.show()
