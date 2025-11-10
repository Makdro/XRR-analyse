import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

# Import des trois modules (assure-toi qu’ils sont dans le même dossier)
from xrr_auto_dash import run_auto
from xrr_manual_dash import run_manual
from xrr_thinlayer_dash import run_thinlayer

st.set_page_config(page_title="XRR Analysis", layout="wide")

st.title("📊 XRR Analysis Tool")
st.markdown("Analyse automatique ou manuelle des mesures XRR (réflectométrie aux rayons X).")

# ===============================================================
# 🗂️ Upload du fichier
# ===============================================================
uploaded_file = st.file_uploader("Dépose ton fichier .xy ici", type=["xy", "txt"])

if uploaded_file:
    # Lecture du fichier
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    data = np.loadtxt(stringio)
    two_theta = data[:, 0]
    intensity = data[:, 1]

    # Affichage signal brut
    st.subheader("Signal brut")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(two_theta, intensity, color='blue')
    ax.set_yscale("log")
    ax.set_xlabel("2θ (deg)")
    ax.set_ylabel("Intensité (a.u.) [log]")
    ax.grid(True, ls='--')
    st.pyplot(fig)

    # ===============================================================
    # ⚙️ Choix du mode d’analyse
    # ===============================================================
    st.subheader("Choix du mode d'analyse")
    mode = st.radio(
        "Sélectionne ton mode :",
        ["Auto", "Manual", "Thin Layer"]
    )

    # ===============================================================
    # 🚀 Lancement de l’analyse
    # ===============================================================
    if st.button("Lancer l'analyse"):
        with st.spinner("Analyse en cours..."):
            try:
                if mode == "Auto":
                    t, rho_mass, peaks = run_auto(two_theta, intensity)
                    st.success(f"✅ Épaisseur : {t:.2f} nm — Densité : {rho_mass:.3f} g/cm³")

                elif mode == "Manual":
                    st.info("Clique sur le graphique ci-dessous pour sélectionner les oscillations.")
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(two_theta, intensity, color='blue')
                    ax.set_yscale("log")
                    ax.set_xlabel("2θ (deg)")
                    ax.set_ylabel("Intensité (a.u.) [log]")
                    ax.grid(True, ls='--')
                    st.pyplot(fig)

                    theta_input = st.text_input("Entre les positions 2θ des pics séparées par des virgules (ex: 0.9, 1.3, 1.7)")
                    if theta_input:
                        theta_peaks = [float(x.strip()) for x in theta_input.split(",")]
                        results = run_manual(two_theta, intensity, theta_peaks)
                        st.success(f"✅ Épaisseur estimée : {results['t']:.2f} nm (R²={results['r2']:.4f})")

                elif mode == "Thin Layer":
                    theta_input = st.text_input("Entre les positions 2θ des pics pour la couche mince (ex: 0.9, 1.3, 1.7)")
                    if theta_input:
                        theta_peaks = [float(x.strip()) for x in theta_input.split(",")]
                        results = run_thinlayer(two_theta, intensity, theta_peaks)
                        st.success(f"✅ Épaisseur estimée : {results['t']:.2f} nm (R²={results['r2']:.4f})")

            except Exception as e:
                st.error(f"Erreur : {e}")
