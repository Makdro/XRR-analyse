# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 14:14:04 2025

@author: Maikane DEROO
"""


def run_thinlayer(two_theta, intensity, theta_peaks):
    """
    Calcule l'épaisseur d'après des pics sélectionnés manuellement
    (Thin Layer version, sans interaction graphique).

    Parameters
    ----------
    two_theta : array-like
        Angles 2θ (degrés)
    intensity : array-like
        Intensités mesurées
    theta_peaks : array-like
        Liste des positions (degrés) des pics sélectionnés par l'utilisateur

    Returns
    -------
    dict contenant :
        - 't' : épaisseur (nm)
        - 'a' : coefficient du fit
        - 'r2' : coefficient de détermination
        - 'theta_peaks' : pics utilisés
        - 'fit_curve' : m² et θ²_fit (pour affichage)
        - 'm2', 'theta2' : données brutes
    """

    import numpy as np
    from sklearn.linear_model import LinearRegression

    # Constantes physiques
    lambda_Cu = 0.15418  # nm (Cu Kα)

    theta_peaks = np.array(sorted(theta_peaks))
    if len(theta_peaks) < 2:
        raise ValueError("Il faut au moins deux oscillations pour calculer l'épaisseur.")

    # ===============================================================
    # 1️⃣ Calcul de θ² et m²
    # ===============================================================
    theta_rad = np.deg2rad(theta_peaks / 2)
    m = np.arange(1, len(theta_rad) + 1)
    m2 = m ** 2
    theta2 = theta_rad ** 2

    # ===============================================================
    # 2️⃣ Fit linéaire θ² = a·m² + b
    # ===============================================================
    reg = LinearRegression().fit(m2.reshape(-1, 1), theta2)
    a = reg.coef_[0]
    b = reg.intercept_
    r2 = reg.score(m2.reshape(-1, 1), theta2)
    theta2_fit = reg.predict(m2.reshape(-1, 1))

    # ===============================================================
    # 3️⃣ Épaisseur
    # ===============================================================
    t = np.sqrt(lambda_Cu ** 2 / (4 * a))

    # ===============================================================
    # 4️⃣ Préparation du retour pour ton app
    # ===============================================================
    results = {
        "t": t,
        "a": a,
        "r2": r2,
        "theta_peaks": theta_peaks,
        "fit_curve": {"m2": m2, "theta2_fit": theta2_fit},
        "m2": m2,
        "theta2": theta2,
    }

    return results
