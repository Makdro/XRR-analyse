# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 14:34:27 2025

@author: Maikane DEROO
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_filter1d

# ===============================================================
# 🔧 PARAMÈTRES
# ===============================================================
file_path = r"C:/Users/Maikane DEROO/Desktop/Voltify/EXPERIENCES/XRR/XRR 28-10-2025/Al2O3 + développeur AZ326MIF/2509RE_Ref09_al2o3.xy"
lambda_Cu = 0.15418  # nm (Cu Kα)

theta_min = 0.2
theta_max = 4.0
zoom_min = 0.5
zoom_max = 4  # ajuster selon ton signal
figsize_x, figsize_y = 15, 5
sigma_smooth = 5  # pour le détendage

# ===============================================================
# 1️⃣ Chargement des données
# ===============================================================
data = np.loadtxt(file_path)
two_theta = data[:, 0]
intensity = data[:, 1]

mask = (two_theta >= theta_min) & (two_theta <= theta_max)
two_theta_zone = two_theta[mask]
intensity_zone = intensity[mask]

# ===============================================================
# 2️⃣ Détendage du signal pour mieux voir les oscillations
# ===============================================================
intensity_smooth = gaussian_filter1d(intensity_zone, sigma=sigma_smooth)
intensity_detrend = intensity_zone / intensity_smooth  # oscillations autour de 1

# ===============================================================
# 3️⃣ Affichage interactif pour sélectionner les pics
# ===============================================================
plt.figure(figsize=(figsize_x, figsize_y))
plt.plot(two_theta_zone, intensity_detrend, label='Signal détendré')
plt.xlabel('2θ (deg)')
plt.ylabel('Signal détendré')
plt.grid(True, which='both', ls='--')
plt.title("Clique sur les premières oscillations, puis ferme la fenêtre")
plt.legend()
plt.xlim(zoom_min, zoom_max)

theta_peaks = plt.ginput(n=-1, timeout=0)  # clique autant de pics que tu veux
theta_peaks = np.array(sorted([x[0] for x in theta_peaks]))
plt.close()

if len(theta_peaks) < 2:
    raise ValueError("Il faut au moins deux oscillations pour calculer l'épaisseur.")

print(f"\n✅ {len(theta_peaks)} oscillations sélectionnées : {theta_peaks}")

# ===============================================================
# 4️⃣ Calcul de l’épaisseur
# ===============================================================
theta_rad = np.deg2rad(theta_peaks / 2)
m = np.arange(1, len(theta_rad) + 1)
m2 = m**2
theta2 = theta_rad**2

reg = LinearRegression().fit(m2.reshape(-1, 1), theta2)
a = reg.coef_[0]
b = reg.intercept_
r2 = reg.score(m2.reshape(-1,1), theta2)
theta2_fit = reg.predict(m2.reshape(-1,1))

t = np.sqrt(lambda_Cu**2 / (4 * a))

# ===============================================================
# 5️⃣ Résultats
# ===============================================================
print("\n==============================")
print(f"Épaisseur estimée : {t:.2f} nm")
print(f"Coefficient a = {a:.3e}")
print(f"R² = {r2:.4f}")
print("==============================")

# ===============================================================
# 6️⃣ Graphiques finaux
# ===============================================================
# Fit θ² vs m²
plt.figure(figsize=(8,5))
plt.plot(m2, theta2, 'o', label='Données saisies')
plt.plot(m2, theta2_fit, '-', label=f'Fit : θ² = {a:.3e}·m² + {b:.3e}, R²={r2:.4f}')
plt.xlabel('m²')
plt.ylabel('θ² (rad²)')
plt.title('Fit θ² vs m²')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Signal détendré avec oscillations sélectionnées
plt.figure(figsize=(figsize_x, figsize_y))
plt.plot(two_theta_zone, intensity_detrend, label='Signal détendré')
plt.scatter(theta_peaks, np.interp(theta_peaks, two_theta_zone, intensity_detrend),
            color='red', label='Oscillations sélectionnées')
plt.xlabel('2θ (deg)')
plt.ylabel('Signal détendré')
plt.title(f'Signal XRR avec oscillations sélectionnées\nÉpaisseur ≈ {t:.2f} nm')
plt.grid(True, which='both', ls='--')
plt.legend()
plt.tight_layout()
plt.show()
