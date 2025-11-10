# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 14:34:27 2025

@author: Maikane DEROO
"""


def run_manual(two_theta, intensity, theta_peaks):
    """
    Calcule l'épaisseur d'après des pics choisis manuellement (par clic ou sélection dans l'UI).

    Parameters
    ----------
    two_theta : array-like
        Angles 2θ du signal XRR (en degrés)
    intensity : array-like
        Intensité correspondante
    theta_peaks : array-like
        Liste des positions (en degrés) des pics sélectionnés manuellement

    Returns
    -------
    dict contenant :
        - 't' : épaisseur (nm)
        - 'a' : coefficient du fit
        - 'r2' : coefficient de détermination R²
        - 'fit_curve' : points ajustés θ² vs m² (pour affichage)
        - 'theta2' : valeurs θ² observées
        - 'm2' : indices m²
    """

    import numpy as np
    from sklearn.linear_model import LinearRegression

    # Constantes physiques
    lambda_Cu = 0.15418  # nm

    theta_peaks = np.array(sorted(theta_peaks))
    if len(theta_peaks) < 2:
        raise ValueError("Il faut au moins deux oscillations pour calculer l'épaisseur.")

    # ===============================================================
    # 1️⃣ Conversion et préparation
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
    # 3️⃣ Épaisseur à partir de la pente
    # ===============================================================
    t = np.sqrt(lambda_Cu ** 2 / (4 * a))

    # ===============================================================
    # 4️⃣ Préparation des résultats (pour Dash)
    # ===============================================================
    results = {
        "t": t,
        "a": a,
        "r2": r2,
        "fit_curve": {"m2": m2, "theta2_fit": theta2_fit},
        "theta2": theta2,
        "m2": m2,
    }

    return results

