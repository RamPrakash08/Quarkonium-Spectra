// Kandarp Solanki 210260026

#include <iostream>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <fstream>

using namespace std;

const double hbar_c = 197.3269804;
const double m = 938.2720813 / 2.0;
const double V0 = -40.0;
const double b = 1.895;
const double c = 0.4;
const int N = 1000;
const double r_max = 7.0;

double random_number(double min_val = 0.0, double max_val = 1.0) {
    return min_val + (max_val - min_val) * rand() / RAND_MAX;
}

double potential(double r) {
    if (r < (b + c)) {
        return V0;
    }
    return 0.0;
}

double trial_wavefunction(double r, double alpha) {
    if (r <= c) {
        return 0.0;
    } else {
        return (r - c) * exp(-alpha * r);
    }
}

double hamiltonian(double r, double alpha) {
    double laplacian = -(hbar_c * hbar_c / (2 * m)) * (pow(alpha, 2) * (r - c) - 2 * alpha) / max(r - c, 1e-5);
    return laplacian + potential(r);
}

double normalize_wavefunction(double alpha) {
    double integral = 0.0;
    double r1, r2, dr = 0.01;

    r1 = c;
    r2 = b + c;
    for (double r = r1; r < r2; r += dr) {
        double psi_r = trial_wavefunction(r, alpha);
        integral += psi_r * psi_r * dr;
    }

    r1 = b + c;
    r2 = r_max;
    for (double r = r1; r < r2; r += dr) {
        double psi_r = trial_wavefunction(r, alpha);
        integral += psi_r * psi_r * dr;
    }

    return 1.0 / sqrt(integral);
}

double monte_carlo_integration(double alpha) {
    double total_energy = 0.0;
    double normalization_constant = normalize_wavefunction(alpha);
    double r = random_number(c, b + c);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double r_new = r + random_number(-0.5, 0.5);
            r_new = max(r_new, c);
            r_new = min(r_new, r_max);

            double psi_r = trial_wavefunction(r, alpha);
            double psi_r_new = trial_wavefunction(r_new, alpha);
            double p = pow(psi_r_new / max(psi_r, 1e-10), 2);

            if (random_number(0.0, 1.0) < p) {
                r = r_new;
            }
        }

        double psi = trial_wavefunction(r, alpha);
        if (psi != 0.0) {
            double energy = hamiltonian(r, alpha);
            total_energy += energy;
        }
    }

    return total_energy / N;
}

pair<double, double> grid_search_minimization(const vector<double>& alpha_range) {
    double min_energy = numeric_limits<double>::infinity();
    double optimal_alpha = 0.5;

    for (double alpha : alpha_range) {
        double energy = monte_carlo_integration(alpha);
        if (energy < min_energy) {
            min_energy = energy;
            optimal_alpha = alpha;
        }
    }

    return {optimal_alpha, min_energy};
}

int main() {
    srand(static_cast<unsigned>(time(0)));

    vector<double> alpha_range(200);
    for (int i = 0; i < alpha_range.size(); ++i) {
        alpha_range[i] = 0.005 * i;
    }

    auto [optimal_alpha, ground_state_energy] = grid_search_minimization(alpha_range);
    cout << "Optimal alpha: " << optimal_alpha << "\n";
    cout << "Ground state energy: " << ground_state_energy << " MeV\n";

    double E_analytic = -2.225;
    double alpha_analytic = sqrt(-2.0 * m * E_analytic) / hbar_c;
    double k_analytic = sqrt(2.0 * m * (E_analytic - V0)) / hbar_c;
    double A = sqrt(2.0 * alpha_analytic / (1 + alpha_analytic * b));
    double B = sqrt((2.0 * alpha_analytic * sin(k_analytic * b) * exp(2 * alpha_analytic * (b + c))) / (1 + alpha_analytic * b));

    cout << "Analytic alpha: " << alpha_analytic << "\n";

    // ofstream outputFile("wavefunction.txt");
    // double normalization_constant = normalize_wavefunction(optimal_alpha);
    // for (double r = c; r < b + c + 10.0; r += 0.1) {
    //     double psi_mc = 2.0 * pow(optimal_alpha, 1.5) * trial_wavefunction(r, optimal_alpha);
    //     double psi_analytic = (r < b + c) ? A * sin(k_analytic * (r - c)) : B * exp(-alpha_analytic * r);
    //     outputFile << r << " " << psi_mc << " " << psi_analytic << "\n";
    // }
    // outputFile.close();

    return 0;
}
