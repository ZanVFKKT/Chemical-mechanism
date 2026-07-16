# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 12:44:40 2026

@author: TilenK
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

# ==========================================================
# PARAMETERS
# ==========================================================

a = 2.040
b = 0.835
c = 16.647
d = 0.0420

I = 2.054
sign = 1.0

# H2O2 parameter values

h_values = [10]

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

def system(t, y, h):

    theta_i, phi_i, theta_j, phi_j = y

    s_i = 1.0 - theta_i - phi_i
    s_j = 1.0 - theta_j - phi_j

    dtheta_i = (
        a*h*s_i
        - b*theta_i
        - c*h*theta_i*s_i**2
        - theta_i*s_i
    )

    dtheta_j = (
        a*h*s_j
        - b*theta_j
        - c*h*theta_j*s_j**2
        - theta_j*s_j
    )

    dphi_i = (
        theta_i*s_i
        - d*h*phi_i
        + sign*I*dtheta_j
    )

    dphi_j = (
        theta_j*s_j
        - d*h*phi_j
        + sign*I*dtheta_i
    )

    return [
        dtheta_i,
        dphi_i,
        dtheta_j,
        dphi_j
    ]

# ==========================================================
# TIME
# ==========================================================

t_start = 0
t_end = 400

dt = 0.02

t_eval = np.arange(
    t_start,
    t_end + dt,
    dt
)

transient_time = 200

# ==========================================================
# PLOT
# ==========================================================

plt.figure(figsize=(11,6))

print()
print("="*85)
print(
    f"{'h':>6}"
    f"{'Amplitude':>15}"
    f"{'Period':>15}"
    f"{'Width90%':>15}"
    f"{'Max slope':>15}"
)
print("="*85)

for h in h_values:

    sol = solve_ivp(
        lambda t,y: system(t,y,h),
        (t_start,t_end),
        y0,
        method="BDF",
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10,
        max_step=0.2
    )

    t = sol.t

    theta = sol.y[0]

    # ------------------------------------------------------

    mask = (t >= 300) & (t <= 320)

    t2 = t[mask]
    theta2 = theta[mask]

    # ------------------------------------------------------
    # Amplitude
    # ------------------------------------------------------

    theta_max = np.max(theta2)
    theta_min = np.min(theta2)

    amplitude = theta_max - theta_min

    # ------------------------------------------------------
    # Peaks
    # ------------------------------------------------------

    peaks, _ = find_peaks(
        theta2,
        prominence=0.00000000001,
        distance=int(2/dt)
    )

    peak_times = t2[peaks]

    if len(peak_times) > 1:

        period = np.mean(np.diff(peak_times))

    else:

        period = np.nan

    # ------------------------------------------------------
    # Peak width (90 %)
    # ------------------------------------------------------

    threshold = theta_min + 0.9*amplitude

    above = theta2 > threshold

    widths = []

    inside = False

    start = None

    for i in range(len(above)):

        if above[i] and not inside:

            inside = True

            start = t2[i]

        elif (not above[i]) and inside:

            inside = False

            stop = t2[i]

            widths.append(stop-start)

    if len(widths):

        width = np.mean(widths)

    else:

        width = np.nan

    # ------------------------------------------------------
    # Maximum slope
    # ------------------------------------------------------

    dtheta = np.gradient(theta2,t2)

    max_slope = np.max(dtheta)

    # ------------------------------------------------------

    print(
        f"{h:6.1f}"
        f"{amplitude:15.4f}"
        f"{period:15.4f}"
        f"{width:15.4f}"
        f"{max_slope:15.4f}"
    )

    plt.plot(
        t2,
        theta2,
        linewidth=2,
        label=f"h = {h}"
    )

# ==========================================================
# PLOT
# ==========================================================

plt.xlabel(r'$\tau$',fontsize=14)

plt.ylabel(r'$\theta_i$',fontsize=14)

plt.title(
    r'Post-transient evolution of $\theta_i$',
    fontsize=15
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()