import math
import random
import numpy as np

hbar_c = 197.3269804
m = 938.2720813 / 2.0
V0 = -40.0
b = 1.895
c = 0.4
N = 1000
r_max = 7.0

def random_number(min_val=0.0, max_val=1.0):
    return min_val + (max_val - min_val) * random.random()

def potential(r):
    if r < (b + c):
        return V0
    return 0.0

def trial_wavefunction(r, alpha):
    if r <= c:
        return 0.0
    else:
        return (r - c) * math.exp(-alpha * r)

def hamiltonian(r, alpha):
    laplacian = -(hbar_c ** 2 / (2 * m)) * (alpha ** 2 * (r - c) - 2 * alpha) / max(r - c, 1e-5)
    return laplacian + potential(r)

def normalize_wavefunction(alpha):
    integral = 0.0
    dr = 0.01

    r1 = c
    r2 = b + c
    for r in np.arange(r1, r2, dr):
        psi_r = trial_wavefunction(r, alpha)
        integral += psi_r ** 2 * dr

    r1 = b + c
    r2 = r_max
    for r in np.arange(r1, r2, dr):
        psi_r = trial_wavefunction(r, alpha)
        integral += psi_r ** 2 * dr

    return 1.0 / math.sqrt(integral)

def monte_carlo_integration(alpha):
    total_energy = 0.0
    normalization_constant = normalize_wavefunction(alpha)
    r = random_number(c, b + c)

    for _ in range(N):
        for _ in range(N):
            r_new = r + random_number(-0.5, 0.5)
            r_new = max(r_new, c)
            r_new = min(r_new, r_max)

            psi_r = trial_wavefunction(r, alpha)
            psi_r_new = trial_wavefunction(r_new, alpha)
            p = (psi_r_new / max(psi_r, 1e-10)) ** 2

            if random_number(0.0, 1.0) < p:
                r = r_new

        psi = trial_wavefunction(r, alpha)
        if psi != 0.0:
            energy = hamiltonian(r, alpha)
            total_energy += energy

    return total_energy / N

def grid_search_minimization(alpha_range):
    min_energy = float('inf')
    optimal_alpha = 0.5
    
    for alpha in alpha_range:
        energy = monte_carlo_integration(alpha)
        if energy < min_energy:
            min_energy = energy
            optimal_alpha = alpha
    
    return optimal_alpha, min_energy

def main():
    random.seed()
    
    alpha_range = [0.005 * i for i in range(200)]
    
    optimal_alpha, ground_state_energy = grid_search_minimization(alpha_range)
    print(f"Optimal alpha: {optimal_alpha}")
    print(f"Ground state energy: {ground_state_energy} MeV")
    
    E_analytic = -2.225
    alpha_analytic = math.sqrt(-2.0 * m * E_analytic) / hbar_c
    k_analytic = math.sqrt(2.0 * m * (E_analytic - V0)) / hbar_c
    A = math.sqrt(2.0 * alpha_analytic / (1 + alpha_analytic * b))
    B = math.sqrt((2.0 * alpha_analytic * math.sin(k_analytic * b) * math.exp(2 * alpha_analytic * (b + c))) / (1 + alpha_analytic * b))

    print(f"Analytic alpha: {alpha_analytic}")
    
    # with open("wavefunction.txt", "w") as outputFile:
    #     normalization_constant = normalize_wavefunction(optimal_alpha)
    #     for r in np.arange(c, b + c + 10.0, 0.1):
    #         psi_mc = 2.0 * optimal_alpha ** 1.5 * trial_wavefunction(r, optimal_alpha)
    #         psi_analytic = A * math.sin(k_analytic * (r - c)) if r < b + c else B * math.exp(-alpha_analytic * r)
    #         outputFile.write(f"{r} {psi_mc} {psi_analytic}\n")

if __name__ == "__main__":
    main()
