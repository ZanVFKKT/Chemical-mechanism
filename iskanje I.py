# -*- coding: utf-8 -*-
"""
Iskanje parametra I za določene koncentracije H2O2 (parameter h). Išče tako,
da poskuša dobiti oscilacije za znane frekvence - exp vrednosti.
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.signal import find_peaks


# ==========================================================
# FIXED MODEL PARAMETERS
# ==========================================================

a = 2.040
b = 0.835
c = 16.647
d = 0.0420

sign = 1.0

# Calibrated dimensional rate constant
k3 = 0.745  # min^-1


# ==========================================================
# EXPERIMENTAL DATA
# ==========================================================
#
# h = 10 corresponds approximately to 0.6% H2O2.
#
# The final experimental frequency at h = 10 is based on
# the measured time series at 0.6% H2O2.
# Modify it if you want to use a different exact value.
# ==========================================================

h_values = np.array([
    5.983,
    6.267,
    7.083,
    7.617,
    8.167,
    10.000
])

frequencies_exp = np.array([
    0.0915,
    0.1075,
    0.1264,
    0.1394,
    0.1977,
    0.2130
])  # min^-1


# ==========================================================
# INITIAL CONDITIONS
# ==========================================================

theta_i0 = 0.5255
phi_i0 = 0.2630

theta_j0 = 0.5793
phi_j0 = 0.2630

y0 = [
    theta_i0,
    phi_i0,
    theta_j0,
    phi_j0
]


# ==========================================================
# DIFFERENTIAL EQUATIONS
# ==========================================================

def system(t, y, h, I):
    theta_i, phi_i, theta_j, phi_j = y

    s_i = 1.0 - theta_i - phi_i
    s_j = 1.0 - theta_j - phi_j

    dtheta_i = (
        a * h * s_i
        - b * theta_i
        - c * h * theta_i * s_i**2
        - theta_i * s_i
    )

    dtheta_j = (
        a * h * s_j
        - b * theta_j
        - c * h * theta_j * s_j**2
        - theta_j * s_j
    )

    dphi_i = (
        theta_i * s_i
        - d * h * phi_i
        + sign * I * dtheta_j
    )

    dphi_j = (
        theta_j * s_j
        - d * h * phi_j
        + sign * I * dtheta_i
    )

    return [
        dtheta_i,
        dphi_i,
        dtheta_j,
        dphi_j
    ]


# ==========================================================
# TIME SETTINGS
# ==========================================================

t_start = 0.0
t_end = 500.0

dt_output = 0.05

t_eval = np.arange(
    t_start,
    t_end + dt_output,
    dt_output
)

# Period is measured in this interval
analysis_start = 300.0

# Two late intervals used to test whether oscillations decay
amplitude_window_1 = (300.0, 400.0)
amplitude_window_2 = (400.0, 500.0)


# ==========================================================
# OSCILLATION-DETECTION SETTINGS
# ==========================================================

relative_prominence = 0.10
minimum_absolute_prominence = 1e-10

minimum_peak_distance_time = 1.0

minimum_peak_distance_points = max(
    1,
    int(minimum_peak_distance_time / dt_output)
)

# Minimum amplitude required for a physically meaningful
# oscillation. Change if needed.
minimum_amplitude = 1e-4

# Late amplitude must remain at least this fraction of the
# preceding amplitude to be regarded as sustained.
minimum_amplitude_ratio = 0.80

# Minimum number of detected peaks
minimum_number_of_peaks = 4


# ==========================================================
# SEARCH RANGE FOR I
# ==========================================================

I_min = 0.5
I_max = 3.5

# First, coarse search
I_coarse_step = 0.05

# Then, refined search around the best coarse value
I_fine_half_width = 0.08
I_fine_step = 0.002

I_coarse_values = np.arange(
    I_min,
    I_max + I_coarse_step,
    I_coarse_step
)


# ==========================================================
# FUNCTION FOR ONE SIMULATION
# ==========================================================

def simulate_and_analyse(h, I):
    """
    Simulate the model and return oscillation properties.

    Returns a dictionary containing:
        valid
        tau_period
        model_frequency
        amplitude
        amplitude_ratio
        number_of_peaks
        solution
    """

    solution = solve_ivp(
        fun=lambda t, y: system(t, y, h, I),
        t_span=(t_start, t_end),
        y0=y0,
        method="BDF",
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10,
        max_step=0.2
    )

    if not solution.success:
        return {
            "valid": False,
            "tau_period": np.nan,
            "model_frequency": np.nan,
            "amplitude": np.nan,
            "amplitude_ratio": np.nan,
            "number_of_peaks": 0,
            "solution": None
        }

    t = solution.t
    theta = solution.y[0]

    # ------------------------------------------------------
    # AMPLITUDE DECAY TEST
    # ------------------------------------------------------

    mask_window_1 = (
        (t >= amplitude_window_1[0])
        & (t <= amplitude_window_1[1])
    )

    mask_window_2 = (
        (t >= amplitude_window_2[0])
        & (t <= amplitude_window_2[1])
    )

    amplitude_1 = np.ptp(theta[mask_window_1])
    amplitude_2 = np.ptp(theta[mask_window_2])

    if amplitude_1 > 0.0:
        amplitude_ratio = amplitude_2 / amplitude_1
    else:
        amplitude_ratio = 0.0

    # Use the final window for the main amplitude
    amplitude = amplitude_2

    # ------------------------------------------------------
    # PERIOD DETECTION
    # ------------------------------------------------------

    analysis_mask = t >= analysis_start

    t_analysis = t[analysis_mask]
    theta_analysis = theta[analysis_mask]

    prominence_value = max(
        relative_prominence * np.ptp(theta_analysis),
        minimum_absolute_prominence
    )

    peaks, _ = find_peaks(
        theta_analysis,
        prominence=prominence_value,
        distance=minimum_peak_distance_points
    )

    number_of_peaks = len(peaks)

    if number_of_peaks >= 2:
        peak_times = t_analysis[peaks]
        periods = np.diff(peak_times)

        tau_period = np.mean(periods)
        model_frequency = k3 / tau_period
    else:
        tau_period = np.nan
        model_frequency = np.nan

    # ------------------------------------------------------
    # VALIDITY TEST
    # ------------------------------------------------------

    valid = (
        number_of_peaks >= minimum_number_of_peaks
        and amplitude >= minimum_amplitude
        and amplitude_ratio >= minimum_amplitude_ratio
        and np.isfinite(tau_period)
        and tau_period > 0.0
    )

    return {
        "valid": valid,
        "tau_period": tau_period,
        "model_frequency": model_frequency,
        "amplitude": amplitude,
        "amplitude_ratio": amplitude_ratio,
        "number_of_peaks": number_of_peaks,
        "solution": solution
    }


# ==========================================================
# SEARCH FOR OPTIMAL I
# ==========================================================

optimal_I = np.full(len(h_values), np.nan)
optimal_tau_period = np.full(len(h_values), np.nan)
optimal_model_frequency = np.full(len(h_values), np.nan)
optimal_amplitude = np.full(len(h_values), np.nan)
optimal_amplitude_ratio = np.full(len(h_values), np.nan)
optimal_number_of_peaks = np.zeros(len(h_values), dtype=int)
frequency_error = np.full(len(h_values), np.nan)

best_solutions = {}


print()
print("Searching for optimal I values...")
print()


for index, (h, frequency_target) in enumerate(
    zip(h_values, frequencies_exp)
):

    print(
        f"Processing h = {h:.3f}, "
        f"target frequency = {frequency_target:.4f} min^-1"
    )

    # ------------------------------------------------------
    # COARSE SEARCH
    # ------------------------------------------------------

    best_coarse_I = np.nan
    best_coarse_error = np.inf
    best_coarse_result = None

    for I in I_coarse_values:

        result = simulate_and_analyse(h, I)

        if not result["valid"]:
            continue

        error = abs(
            result["model_frequency"]
            - frequency_target
        )

        if error < best_coarse_error:
            best_coarse_error = error
            best_coarse_I = I
            best_coarse_result = result

    if not np.isfinite(best_coarse_I):
        print(
            "  No sustained oscillatory solution found "
            "in the selected I range."
        )
        print()
        continue

    # ------------------------------------------------------
    # FINE SEARCH
    # ------------------------------------------------------

    fine_start = max(
        I_min,
        best_coarse_I - I_fine_half_width
    )

    fine_end = min(
        I_max,
        best_coarse_I + I_fine_half_width
    )

    I_fine_values = np.arange(
        fine_start,
        fine_end + I_fine_step,
        I_fine_step
    )

    best_I = best_coarse_I
    best_error = best_coarse_error
    best_result = best_coarse_result

    for I in I_fine_values:

        result = simulate_and_analyse(h, I)

        if not result["valid"]:
            continue

        error = abs(
            result["model_frequency"]
            - frequency_target
        )

        if error < best_error:
            best_error = error
            best_I = I
            best_result = result

    # ------------------------------------------------------
    # STORE BEST RESULT
    # ------------------------------------------------------

    optimal_I[index] = best_I
    optimal_tau_period[index] = best_result["tau_period"]
    optimal_model_frequency[index] = best_result["model_frequency"]
    optimal_amplitude[index] = best_result["amplitude"]
    optimal_amplitude_ratio[index] = best_result["amplitude_ratio"]
    optimal_number_of_peaks[index] = best_result["number_of_peaks"]

    frequency_error[index] = (
        best_result["model_frequency"]
        - frequency_target
    )

    best_solutions[h] = best_result["solution"]

    print(f"  Optimal I            = {best_I:.4f}")
    print(
        f"  Model frequency      = "
        f"{best_result['model_frequency']:.6f} min^-1"
    )
    print(
        f"  Experimental freq.   = "
        f"{frequency_target:.6f} min^-1"
    )
    print(
        f"  Absolute error       = "
        f"{best_error:.6f} min^-1"
    )
    print(
        f"  Dimensionless period = "
        f"{best_result['tau_period']:.6f}"
    )
    print(
        f"  Final amplitude      = "
        f"{best_result['amplitude']:.6e}"
    )
    print(
        f"  Amplitude ratio      = "
        f"{best_result['amplitude_ratio']:.4f}"
    )
    print()


# ==========================================================
# PRINT FINAL TABLE
# ==========================================================

print()
print("=" * 130)

print(
    f"{'h':>9}"
    f"{'f exp':>13}"
    f"{'I optimal':>14}"
    f"{'tau period':>15}"
    f"{'f model':>15}"
    f"{'error':>14}"
    f"{'amplitude':>15}"
    f"{'amp ratio':>14}"
    f"{'peaks':>10}"
)

print("=" * 130)

for index in range(len(h_values)):

    print(
        f"{h_values[index]:9.3f}"
        f"{frequencies_exp[index]:13.6f}"
        f"{optimal_I[index]:14.6f}"
        f"{optimal_tau_period[index]:15.6f}"
        f"{optimal_model_frequency[index]:15.6f}"
        f"{frequency_error[index]:14.6f}"
        f"{optimal_amplitude[index]:15.6e}"
        f"{optimal_amplitude_ratio[index]:14.6f}"
        f"{optimal_number_of_peaks[index]:10d}"
    )

print("=" * 130)


# ==========================================================
# PLOT OPTIMAL I VERSUS h
# ==========================================================

valid = np.isfinite(optimal_I)

plt.figure(figsize=(9, 6))

plt.plot(
    h_values[valid],
    optimal_I[valid],
    marker="o",
    linewidth=2
)

plt.xlabel(
    r"Dimensionless H$_2$O$_2$ concentration, $h$",
    fontsize=13
)

plt.ylabel(
    r"Optimal coupling parameter, $I$",
    fontsize=13
)

plt.title(
    r"Optimal coupling parameter as a function of $h$"
)

plt.grid(True)
plt.tight_layout()
plt.show()


# ==========================================================
# PLOT MODEL AND EXPERIMENTAL FREQUENCIES
# ==========================================================

plt.figure(figsize=(9, 6))

plt.scatter(
    h_values,
    frequencies_exp,
    s=85,
    label="Experimental data",
    zorder=3
)

plt.plot(
    h_values[valid],
    optimal_model_frequency[valid],
    marker="o",
    linewidth=2,
    label="Model with fitted I"
)

plt.xlabel(
    r"Dimensionless H$_2$O$_2$ concentration, $h$",
    fontsize=13
)

plt.ylabel(
    r"Oscillation frequency (min$^{-1}$)",
    fontsize=13
)

plt.title(
    "Experimental and fitted model frequencies"
)

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ==========================================================
# PLOT BEST TIME SERIES
# ==========================================================

plot_start = 450.0
plot_end = 500.0

plt.figure(figsize=(11, 7))

for h, solution in best_solutions.items():

    mask = (
        (solution.t >= plot_start)
        & (solution.t <= plot_end)
    )

    plt.plot(
        solution.t[mask],
        solution.y[0, mask],
        linewidth=1.5,
        label=rf"$h={h:.3f}$, $I={optimal_I[np.argmin(np.abs(h_values-h))]:.3f}$"
    )

plt.xlabel(r"$\tau$", fontsize=13)
plt.ylabel(r"$\theta_i$", fontsize=13)

plt.title(
    r"Late-time oscillations at fitted values of $I$"
)

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ==========================================================
# SAVE RESULTS
# ==========================================================

output_data = np.column_stack(
    (
        h_values,
        frequencies_exp,
        optimal_I,
        optimal_tau_period,
        optimal_model_frequency,
        frequency_error,
        optimal_amplitude,
        optimal_amplitude_ratio,
        optimal_number_of_peaks
    )
)

header = (
    "h,"
    "experimental_frequency_min-1,"
    "optimal_I,"
    "dimensionless_period_tau,"
    "model_frequency_min-1,"
    "frequency_error_min-1,"
    "final_amplitude,"
    "late_to_early_amplitude_ratio,"
    "number_of_detected_peaks"
)

np.savetxt(
    "optimal_I_for_each_h.csv",
    output_data,
    delimiter=",",
    header=header,
    comments=""
)

print()
print("Results saved to optimal_I_for_each_h.csv")