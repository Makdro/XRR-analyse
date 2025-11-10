# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 09:49:27 2025

@author: Maikane DEROO
"""


def run_auto(two_theta, intensity, num_cycles=1, material_choice="SiO2"):
    import numpy as np
    from scipy.signal import find_peaks, savgol_filter
    from scipy.ndimage import gaussian_filter1d
    from sklearn.linear_model import LinearRegression

    # ===============================================================
    # Paramètres modifiables
    # ===============================================================
    theta_min = 0
    theta_max = 4
    min_theta_ignore = 0.5
    num_oscillations = 10
    sigma_smooth = 5
    window_savgol = 301
    poly_savgol = 3
    lambda_Cu = 0.15418  # nm

    # ===============================================================
    # 2️⃣ Lissage et normalisation
    # ===============================================================
    intensity_smooth = gaussian_filter1d(intensity, sigma=sigma_smooth)
    intensity_detrend = intensity / intensity_smooth

    # ===============================================================
    # 3️⃣ Détection des pics
    # ===============================================================
    signal_log = np.log10(intensity + 1)
    background = savgol_filter(signal_log, window_length=window_savgol, polyorder=poly_savgol)
    signal_detrended = signal_log - background
    signal_smooth = gaussian_filter1d(signal_detrended, sigma=2)
    second_derivative = np.gradient(np.gradient(signal_smooth))

    mask_zone = (two_theta > min_theta_ignore) & (two_theta >= theta_min) & (two_theta <= theta_max)
    idx_zone = np.where(mask_zone)[0]

    if len(idx_zone) == 0:
        raise ValueError("Zone d'analyse vide. Vérifie theta_min/theta_max/min_theta_ignore.")

    abs_sd_max = np.max(np.abs(second_derivative[idx_zone]))
    prom = 1e-3 * max(1e-12, abs_sd_max)
    peaks_all, _ = find_peaks(second_derivative, prominence=prom, distance=2)
    peaks_zone = np.array([p for p in peaks_all if p in idx_zone], dtype=int)

    if len(peaks_zone) < 3:
        local_max_idx = []
        sd = second_derivative
        for i in idx_zone[1:-1]:
            if sd[i] > sd[i - 1] and sd[i] >= sd[i + 1]:
                local_max_idx.append(i)
        peaks_zone = np.array(local_max_idx, dtype=int)

    if peaks_zone.size == 0:
        raise ValueError("Aucun maximum local trouvé dans la zone.")

    # Trier par angle croissant
    peaks_zone = np.array(sorted(peaks_zone, key=lambda ind: two_theta[ind]))
    n = len(peaks_zone)

    if n < num_oscillations:
        peaks_selected = peaks_zone[:n]
    else:
        best_win = None
        best_score = np.inf
        for start in range(0, n - num_oscillations + 1):
            win_inds = peaks_zone[start:start + num_oscillations]
            win_angles = two_theta[win_inds]
            diffs = np.diff(win_angles)
            mean_diff = np.mean(diffs)
            std_diff = np.std(diffs)
            span = win_angles[-1] - win_angles[0]
            score = (
                (win_angles[0] / (theta_max - theta_min + 1e-12)) * 0.7
                + (std_diff / (mean_diff + 1e-12)) * 0.2
                + (span / (theta_max - theta_min + 1e-12)) * 0.1
            )
            if np.any(diffs > 2.5 * np.median(np.abs(np.diff(two_theta[peaks_zone])) + 1e-12)):
                score *= 5.0
            if score < best_score:
                best_score = score
                best_win = win_inds
        peaks_selected = np.array(best_win, dtype=int)

    # ===============================================================
    # 4️⃣ Recentrage sur les vrais maxima
    # ===============================================================
    refined_peaks = []
    window_half_width = 5
    for idx in peaks_selected:
        start = max(0, idx - window_half_width)
        end = min(len(intensity) - 1, idx + window_half_width)
        local_segment = intensity[start:end+1]
        refined_idx = start + np.argmax(local_segment)
        refined_peaks.append(refined_idx)
    peaks_selected = np.array(sorted(set(refined_peaks)), dtype=int)

    # ===============================================================
    # 7️⃣ Fit linéaire θ² vs m² → épaisseur
    # ===============================================================
    theta_rad = np.deg2rad(two_theta[peaks_selected] / 2)
    theta2 = theta_rad ** 2
    m = np.arange(1, len(theta2) + 1)
    m2 = m ** 2
    reg = LinearRegression().fit(m2.reshape(-1, 1), theta2)
    a = reg.coef_[0]
    t = np.sqrt(lambda_Cu ** 2 / (4 * a))

    # ===============================================================
    # 8️⃣ Densité via θc
    # ===============================================================
    from math import pi
    intensity_smooth = gaussian_filter1d(intensity, sigma=10)
    diff_intensity = np.gradient(intensity_smooth, two_theta)
    mask_qc = (two_theta > 0.02) & (two_theta < 2.0)
    idx_local = np.argmin(diff_intensity[mask_qc])
    idx_qc = np.where(mask_qc)[0][idx_local]
    theta_c_rad = np.deg2rad(two_theta[idx_qc] / 2)

    # Constantes
    lambda_A = 1.5418
    r_e = 2.8179403262e-5
    N_A = 6.02214076e23

    materiaux_info = {
        "SiO2": {"M": 60.08, "Z": 30},
        "Al2O3": {"M": 101.96, "Z": 50},
        "TiO2": {"M": 79.87, "Z": 38},
        "Pt": {"M": 195.08, "Z": 78},
    }

    mat = materiaux_info.get(material_choice, {"M": 60.08, "Z": 30})
    M, Z = mat["M"], mat["Z"]

    rho_e = (pi * theta_c_rad**2) / (r_e * lambda_A**2)
    rho_mass = rho_e * M / (Z * N_A) * 1e24

    return t, rho_mass, peaks_selected
