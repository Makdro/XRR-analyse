# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 14:14:04 2025

@author: Maikane DEROO
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ===============================================================
# 🔧 PARAMÈTRES
# ===============================================================
file_path = r"C:/Users/Maikane DEROO/Desktop/Voltify/EXPERIENCES/XRR/XRR 28-10-2025/SiO2 Imane/2509RE01_Ref01_SiO2.xy"
lambda_Cu = 0.15418  # nm (Cu Kα)
theta_min = 0.2
theta_max = 7.0
figsize_x, figsize_y = 15, 5

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
# 2️⃣ Affichage du signal brut et sélection interactive
# ===============================================================
plt.figure(figsize=(figsize_x, figsize_y))
plt.plot(two_theta_zone, intensity_zone, label='Signal brut')
plt.yscale('log')  # échelle log pour voir les oscillations
plt.xlabel('2θ (deg)')
plt.ylabel('Intensité (a.u.) [log]')
plt.grid(True, which='both', ls='--')
plt.title("Clique sur les oscillations (peaks), puis ferme la fenêtre")
plt.legend()

# Sélection interactive
theta_peaks = plt.ginput(n=-1, timeout=0)
theta_peaks = np.array(sorted([x[0] for x in theta_peaks]))
plt.close()

if len(theta_peaks) < 2:
    raise ValueError("Il faut au moins deux oscillations pour calculer l'épaisseur.")

print(f"\n✅ {len(theta_peaks)} oscillations sélectionnées : {theta_peaks}")

# ===============================================================
# 3️⃣ Calcul de l’épaisseur
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
# 4️⃣ Résultats
# ===============================================================
print("\n==============================")
print(f"Épaisseur estimée : {t:.2f} nm")
print(f"Coefficient a = {a:.3e}")
print(f"R² = {r2:.4f}")
print("==============================")

# ===============================================================
# 5️⃣ Graphiques finaux
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

# Signal brut avec oscillations sélectionnées
plt.figure(figsize=(figsize_x, figsize_y))
plt.plot(two_theta_zone, intensity_zone, label='Signal brut')
plt.scatter(theta_peaks, np.interp(theta_peaks, two_theta_zone, intensity_zone),
            color='red', label='Oscillations sélectionnées')
plt.yscale('log')
plt.xlabel('2θ (deg)')
plt.ylabel('Intensité (a.u.) [log]')
plt.title(f'Signal XRR avec oscillations sélectionnées\nÉpaisseur ≈ {t:.2f} nm')
plt.grid(True, which='both', ls='--')
plt.legend()
plt.tight_layout()
plt.show()
