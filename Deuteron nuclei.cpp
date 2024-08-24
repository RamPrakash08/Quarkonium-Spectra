#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <limits>

// Author: Ram Prakash
// Roll Number: 210260042
// Date: 24/08/2024

// Constants
const double HARTREE_TO_EV = 27.2114;
const double MEV_TO_HARTREE = 1.51926741e-5;
const double m_p = 1836.15267343;
const double m_n = 1838.68366173;
const double mu_D = (m_p * m_n) / (m_p + m_n);
const double a0 = 1.0 / mu_D;
const double c = 0.4;  // in atomic units
const double r_21fm = 2.1;  // in atomic units
const double V_40MeV = -40;  // in MeV

// Function prototypes
std::vector<double> potential_energy(const std::vector<double>& r);
std::vector<double> trial_wavefunction(const std::vector<double>& r, double alpha, int p);
double energy_functional(double alpha, int p, const std::vector<double>& r, double dr);
std::pair<double, double> monte_carlo_minimization(const std::vector<double>& r, double dr, int num_steps = 1000, double alpha_init = 0.5, int p = 1);

int main() {
    // Define the range for r
    int num_points = 2000;
    double r_min = 0.01;
    double r_max = 20.0;
    double dr = (r_max - r_min) / (num_points - 1);

    std::vector<double> r(num_points);
    for (int i = 0; i < num_points; ++i) {
        r[i] = r_min + i * dr;
    }

    // Perform Monte Carlo minimization
    auto result = monte_carlo_minimization(r, dr);
    double alpha_opt = result.first;
    double min_energy = result.second;

    // Convert energy to MeV for output
    double min_energy_mev = min_energy;

    std::cout << "Author: Ram Prakash" << std::endl;
    std::cout << "Roll Number: 210260042" << std::endl;
    std::cout << "Date: 24/08/2024" << std::endl;
    std::cout << "Optimal alpha: " << alpha_opt << std::endl;
    std::cout << "Minimum energy: " << min_energy_mev << " MeV" << std::endl;

    return 0;
}

std::vector<double> potential_energy(const std::vector<double>& r) {
    std::vector<double> V(r.size(), 0.0);
    for (size_t i = 0; i < r.size(); ++i) {
        if (r[i] >= c && r[i] <= r_21fm) {
            V[i] = V_40MeV;
        }
    }
    return V;
}

std::vector<double> trial_wavefunction(const std::vector<double>& r, double alpha, int p) {
    std::vector<double> psi(r.size(), 0.0);
    for (size_t i = 0; i < r.size(); ++i) {
        if (r[i] >= c) {
            psi[i] = std::pow(r[i] - c, p) * std::exp(-alpha * (r[i] - c));
        }
    }
    return psi;
}

double energy_functional(double alpha, int p, const std::vector<double>& r, double dr) {
    std::vector<double> psi = trial_wavefunction(r, alpha, p);
    std::vector<double> dpsi_dr(r.size(), 0.0);
    std::vector<double> d2psi_dr2(r.size(), 0.0);

    // Compute first derivative
    for (size_t i = 1; i < r.size() - 1; ++i) {
        dpsi_dr[i] = (psi[i + 1] - psi[i - 1]) / (2 * dr);
    }

    // Compute second derivative
    for (size_t i = 1; i < r.size() - 1; ++i) {
        d2psi_dr2[i] = (psi[i + 1] - 2 * psi[i] + psi[i - 1]) / (dr * dr);
    }

    // Compute kinetic energy
    double kinetic = -0.5 * (std::pow(197.3, 2) / 931.5) * (1 / mu_D);
    for (size_t i = 1; i < r.size() - 1; ++i) {
        kinetic += psi[i] * d2psi_dr2[i] * r[i] * r[i] * dr;
    }

    // Compute potential energy
    std::vector<double> V = potential_energy(r);
    double potential = 0.0;
    for (size_t i = 1; i < r.size(); ++i) {
        potential += V[i] * psi[i] * psi[i] * r[i] * r[i] * dr;
    }

    // Compute normalization
    double normalization = 0.0;
    for (size_t i = 1; i < r.size(); ++i) {
        normalization += psi[i] * psi[i] * r[i] * r[i] * dr;
    }

    double energy = (kinetic + potential) / normalization;
    return energy;
}

std::pair<double, double> monte_carlo_minimization(const std::vector<double>& r, double dr, int num_steps, double alpha_init, int p) {
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 0.01);

    double alpha = alpha_init;
    double best_alpha = alpha;
    double best_energy = energy_functional(alpha, p, r, dr);

    for (int step = 0; step < num_steps; ++step) {
        double new_alpha = alpha + distribution(generator);
        if (new_alpha <= 0) {
            continue;
        }
        double new_energy = energy_functional(new_alpha, p, r, dr);
        if (new_energy < best_energy) {
            best_energy = new_energy;
            best_alpha = new_alpha;
        }
    }

    return {best_alpha, best_energy};
}
