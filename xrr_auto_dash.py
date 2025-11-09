# xrr_manual.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LinearRegression
from io import BytesIO
import base64

def run(filepath, num_cycles=100, material_choice="5", custom_mat=None):
    """
    Lance ton analyse XRR manuelle.
    - filepath : chemin du fichier XRR (.xy ou .txt)
    - num_cycles : nombre de cycles de dépôt
    - material_choice : "1" à "5" pour les matériaux connus, "custom" sinon
    - custom_mat : dict optionnel {"nom":..., "M":..., "Z":...}
    Retourne un dict {epaisseur, densite, message, plot}
    """
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
    # Chargement des données
    # ===============================================================
    data = np.loadtxt(filepath)
    two_theta = data[:, 0]
    intensity = data[:, 1]

    # ===============================================================
    # Lissage et normalisation
    # ===============================================================
    intensity_smooth = gaussian_filter1d(intensity, sigma=sigma_smooth)
    intensity_detrend = intensity / intensity_smooth

    # ===============================================================
    # Détection des pics
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
            if sd[i] > sd[i-1] and sd[i] >= sd[i+1]:
                local_max_idx.append(i)
        peaks_zone = np.array(local_max_idx, dtype=int)

    if peaks_zone.size == 0:
        raise ValueError("Aucun maximum local trouvé dans la zone.")

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
            score = (win_angles[0] / (theta_max - theta_min + 1e-12)) * 0.7 + \
                    (std_diff / (mean_diff + 1e-12)) * 0.2 + \
                    (span / (theta_max - theta_min + 1e-12)) * 0.1
            if np.any(diffs > 2.5 * np.median(np.abs(np.diff(two_theta[peaks_zone])) + 1e-12)):
                score *= 5.0
            if score < best_score:
                best_score = score
                best_win = win_inds
        peaks_selected = np.array(best_win, dtype=int)

    # Recentrage sur les vrais maxima
    refined_peaks = []
    for idx in peaks_selected:
        start = max(0, idx - 5)
        end = min(len(intensity) - 1, idx + 5)
        local_segment = intensity[start:end+1]
        refined_idx = start + np.argmax(local_segment)
        refined_peaks.append(refined_idx)
    peaks_selected = np.array(sorted(set(refined_peaks)), dtype=int)

    # ===============================================================
    # Fit linéaire θ² vs m² → épaisseur
    # ===============================================================
    two_theta_peaks = two_theta[peaks_selected]
    theta_rad = np.deg2rad(two_theta_peaks / 2)
    theta2 = theta_rad**2
    m = np.arange(1, len(theta2)+1)
    m2 = m**2
    reg = LinearRegression().fit(m2.reshape(-1, 1), theta2)
    a = reg.coef_[0]
    r2 = reg.score(m2.reshape(-1,1), theta2)
    t = np.sqrt(lambda_Cu**2 / (4 * a))

    # ===============================================================
    # Densité
    # ===============================================================
    intensity_smooth2 = gaussian_filter1d(intensity, sigma=10)
    diff_intensity = np.gradient(intensity_smooth2, two_theta)
    mask_qc = (two_theta > 0.02) & (two_theta < 2.0)
    idx_qc = np.where(mask_qc)[0][np.argmin(diff_intensity[mask_qc])]
    theta_c_deg_2theta = two_theta[idx_qc]
    theta_c_rad = np.deg2rad(theta_c_deg_2theta / 2.0)

    lambda_A = 1.5418
    r_e = 2.8179403262e-5
    N_A = 6.02214076e23

    materiaux_info = {
        "1": {"nom": "SiO2", "M": 60.08, "Z": 30},
        "2": {"nom": "Al2O3", "M": 101.96, "Z": 50},
        "3": {"nom": "TiO2 (anatase)", "M": 79.87, "Z": 38},
        "4": {"nom": "TiO2 (rutile)", "M": 79.87, "Z": 38},
        "5": {"nom": "Pt", "M": 195.08, "Z": 78}
    }

    if material_choice in materiaux_info:
        mat = materiaux_info[material_choice]
    else:
        mat = custom_mat or {"nom": "Inconnu", "M": 60.0, "Z": 30}

    rho_e = (np.pi * theta_c_rad**2) / (r_e * lambda_A**2)
    rho_mass = rho_e * mat["M"] / (mat["Z"] * N_A) * 1e24

    # ===============================================================
    # Résumé & figures
    # ===============================================================
    GPC_nm = t / num_cycles
    message = f"Épaisseur = {t:.2f} nm, Densité = {rho_mass:.2f} g/cm³, R²={r2:.3f}"

    # Figure reflectivité
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(two_theta, intensity, label="Signal brut", color="blue")
    ax.set_yscale("log")
    ax.set_xlabel("2θ (°)")
    ax.set_ylabel("Intensité (a.u.)")
    ax.grid(True, which='both', ls='--')
    ax.set_title("Signal XRR manuel")
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "epaisseur": float(t),
        "densite": float(rho_mass),
        "message": message,
        "plot": {"img": encoded}
    }
