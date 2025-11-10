# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 09:49:27 2025

@author: Maikane DEROO
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LinearRegression

# ===============================================================
# Paramètres modifiables
# ===============================================================
file_path = 'C:/Users/Maikane DEROO/Desktop/Voltify/EXPERIENCES/XRR/XRR 28-10-2025/Al2O3 + développeur AZ326MIF/2509RE_Ref12_al2o3_2min30.xy'

theta_min = 0        # début de la zone d'analyse
theta_max = 4        # fin de la zone d'analyse
min_theta_ignore = 0.5  # angle à ignorer (artefact initial)
num_oscillations = 10    # nombre de pics à détecter
sigma_smooth = 5          # pour gaussian_filter1d
window_savgol = 301       # pour savgol_filter
poly_savgol = 3

lambda_Cu = 0.15418  # nm

# ===============================================================
# 1️⃣ Chargement des données
# ===============================================================
data = np.loadtxt(file_path)
two_theta = data[:, 0]
intensity = data[:, 1]

plt.figure(figsize=(10,5))
plt.plot(two_theta, intensity, color='blue')
plt.yscale('log')
plt.xlabel('2θ (deg)')
plt.ylabel('Intensity (a.u.) [log scale]')
plt.title('XRR - Signal brut')
plt.grid(True, which='both', ls='--')
plt.show()

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
    # recherche des maxima locaux simples
    local_max_idx = []
    sd = second_derivative
    for i in idx_zone[1:-1]:
        if sd[i] > sd[i-1] and sd[i] >= sd[i+1]:
            local_max_idx.append(i)
    peaks_zone = np.array(local_max_idx, dtype=int)

if peaks_zone.size == 0:
    raise ValueError("Aucun maximum local trouvé dans la zone. Essaie d'élargir theta_max ou diminuer min_theta_ignore.")

# Trier par angle croissant
peaks_zone = np.array(sorted(peaks_zone, key=lambda ind: two_theta[ind]))
theta_peaks_zone = two_theta[peaks_zone]

# Sélection de la fenêtre la plus cohérente
n = len(peaks_zone)
if n < num_oscillations:
    print(f"⚠️ Seulement {n} pics candidats détectés ; on utilisera ceux-ci.")
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
        score = (win_angles[0] / (theta_max - theta_min + 1e-12)) * 0.7 + (std_diff / (mean_diff + 1e-12)) * 0.2 + (span / (theta_max - theta_min + 1e-12)) * 0.1
        if np.any(diffs > 2.5 * np.median(np.abs(np.diff(theta_peaks_zone)) + 1e-12)):
            score *= 5.0
        if score < best_score:
            best_score = score
            best_win = win_inds
    peaks_selected = np.array(best_win, dtype=int)

# ===============================================================
# 4️⃣ Recentrage sur les vrais maxima (signal brut)
# ===============================================================
refined_peaks = []
window_half_width = 5
for idx in peaks_selected:
    start = max(0, idx - window_half_width)
    end = min(len(intensity) - 1, idx + window_half_width)
    local_segment = intensity[start:end+1]
    local_max_rel = np.argmax(local_segment)
    refined_idx = start + local_max_rel
    refined_peaks.append(refined_idx)

refined_peaks = np.array(sorted(set(refined_peaks)), dtype=int)
peaks_selected = refined_peaks
print(f"Recentrage effectué sur {len(refined_peaks)} pics (vrai maximum brut)")

# ===============================================================
# 4️⃣b Affichage des positions des pics
# ===============================================================
print("\n--- Positions des pics sélectionnés ---")
for i, idx in enumerate(peaks_selected, start=1):
    print(f"Pic {i} : 2θ = {two_theta[idx]:.4f}°")


# ===============================================================
# 5️⃣ Visualisation des pics détectés
# ===============================================================
def plot_peaks(two_theta_data, intensity_data, peaks_idx, zoom=None, title=''):
    intensity_detrend_plot = intensity_data / gaussian_filter1d(intensity_data, sigma=sigma_smooth)
    plt.figure(figsize=(10,5))
    plt.plot(two_theta_data, intensity_detrend_plot, label='Signal détendré')
    for i, idx in enumerate(peaks_idx, start=1):
        theta_val = two_theta_data[idx]
        plt.plot(theta_val, intensity_detrend_plot[idx], 'ro')
        plt.text(theta_val, intensity_detrend_plot[idx]*1.2, str(i), color='red', ha='center')
        plt.axvline(theta_val, color='red', linestyle='--', alpha=0.6)
    if zoom:
        plt.xlim(*zoom)
    plt.xlabel('2θ (deg)')
    plt.ylabel('Intensité normalisée')
    plt.title(title)
    plt.grid(True, ls='--')
    plt.legend()
    plt.show()

plot_peaks(two_theta, intensity, peaks_selected, title='Signal complet avec pics détectés')
plot_peaks(two_theta, intensity, peaks_selected, zoom=(theta_min, theta_max),
           title='Zoom sur la zone sélectionnée avec pics')

# ===============================================================
# 6️⃣ Conversion en θ et radians
# ===============================================================
two_theta_peaks = two_theta[peaks_selected]
theta_deg = two_theta_peaks / 2
theta_rad = np.deg2rad(theta_deg)

# ===============================================================
# 7️⃣ Fit linéaire θ² vs m² → épaisseur
# ===============================================================
theta2 = theta_rad**2
m = np.arange(1, len(theta2)+1)
m2 = m**2
reg = LinearRegression().fit(m2.reshape(-1, 1), theta2)
a = reg.coef_[0]
b = reg.intercept_
r2 = reg.score(m2.reshape(-1,1), theta2)
theta2_fit = reg.predict(m2.reshape(-1,1))

plt.figure(figsize=(8,5))
plt.plot(m2, theta2, 'o', label='Données')
plt.plot(m2, theta2_fit, '-', label=f'Fit : θ² = {a:.3e}·m² + {b:.3e}, R²={r2:.4f}')
plt.xlabel('m²')
plt.ylabel('θ² (rad²)')
plt.title('Fit θ² vs m²')
plt.legend()
plt.grid(True)
plt.show()

t = np.sqrt(lambda_Cu**2 / (4 * a))
print(f"\nÉpaisseur de la couche : t = {t:.2f} nm")

# ===============================================================
# 8️⃣ Calcul GPC
# ===============================================================
num_cycles = int(input("\nNombre de cycles de dépôt effectués : "))
GPC_nm = t / num_cycles
GPC_A = GPC_nm * 10
print(f"Vitesse de dépôt : {GPC_nm:.4f} nm/cycle ({GPC_A:.2f} Å/cycle)")

# ===============================================================
# 9️⃣ Détermination de la densité via θc
# ===============================================================
intensity_smooth = gaussian_filter1d(intensity, sigma=10)
diff_intensity = np.gradient(intensity_smooth, two_theta)
mask_qc = (two_theta > 0.02) & (two_theta < 2.0)
idx_local = np.argmin(diff_intensity[mask_qc])
idx_qc = np.where(mask_qc)[0][idx_local]
theta_c_deg_2theta = two_theta[idx_qc]
theta_c_deg = theta_c_deg_2theta / 2.0
theta_c_rad = np.deg2rad(theta_c_deg)

# Constantes
lambda_A = 1.5418  # Å
r_e = 2.8179403262e-5  # Å
N_A = 6.02214076e23    # mol^-1

# Matériaux connus
materiaux_info = {
    "1": {"nom": "SiO2 (amorphe)", "M": 60.08, "Z": 30, "dens": 2.20},
    "2": {"nom": "Al2O3 (amorphe)", "M": 101.96, "Z": 50, "dens": 3.97},
    "3": {"nom": "TiO2 (anatase)", "M": 79.87, "Z": 38, "dens": 3.90},
    "4": {"nom": "TiO2 (rutile)", "M": 79.87, "Z": 38, "dens": 4.23},
    "5": {"nom": "Pt (platine)", "M": 195.08, "Z": 78, "dens": 21.45},
    "6": {"nom": "Autre"}
}

print("\n=== Choisissez le matériau : ===")
for key, mat in materiaux_info.items():
    print(f"{key}. {mat['nom']}")
choix = input("Entrez le numéro correspondant : ")

if choix in ["1","2","3","4","5"]:
    mat = materiaux_info[choix]
    M = mat["M"]
    Z = mat["Z"]
    nom_mat = mat["nom"]
else:
    nom_mat = input("Entrez le nom du matériau : ")
    M = float(input("Entrez la masse molaire M (g/mol) : "))
    Z = int(input("Entrez le nombre d’électrons Z : "))

rho_e = (np.pi * theta_c_rad**2) / (r_e * lambda_A**2)
rho_mass = rho_e * M / (Z * N_A) * 1e24

print("\n=== Densité estimée ===")
print(f"Angle critique détecté : 2θ = {theta_c_deg_2theta:.4f}° → θ = {theta_c_deg:.4f}°")
print(f"θc (rad) = {theta_c_rad:.6e}")
print(f"Densité électronique : ρₑ = {rho_e:.6e} e⁻/Å³")
print(f"Densité massique : ρ = {rho_mass:.3f} g/cm³")

# Interprétation
if rho_mass < 1.0:
    interpretation = "Matériau très poreux ou peu dense"
elif 1.0 <= rho_mass < 2.0:
    interpretation = "Matériau modérément dense"
elif 2.0 <= rho_mass < 4.0:
    interpretation = "Matériau dense (type oxydes amorphes)"
else:
    interpretation = "Matériau très dense (probablement métallique ou compact)"
print(f"Interprétation : {interpretation}")

# Graphique θc
fig, ax1 = plt.subplots(figsize=(10,5))
ax1.plot(two_theta, intensity_smooth, color='blue', label='Signal lissé')
ax1.axvline(2 * theta_c_deg, color='red', linestyle='--', label=f'θc ≈ {theta_c_deg:.3f}°')
ax1.set_yscale('log')
ax1.set_xlabel('2θ (deg)')
ax1.set_ylabel('Intensité (a.u.) [log]')
ax1.legend(loc='upper right')
ax1.grid(True, which='both', ls='--')
ax1.set_title('Détermination de θc et calcul de densité')

ax2 = ax1.twinx()
ax2.plot(two_theta, diff_intensity, color='gray', alpha=0.5, label='Dérivée')
ax2.set_ylabel("dI/d(2θ)")
ax2.legend(loc='lower left')
plt.show()
