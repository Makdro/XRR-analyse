import dash
from dash import Dash, html, dcc, Input, Output, State
import subprocess
import sys
import os

app = dash.Dash(__name__)
app.title = "XRR – Interface complète Voltify"

# ===============================================================
# 🎛️ Layout principal
# ===============================================================
app.layout = html.Div([
    html.H1("Analyse XRR – Voltify", style={'textAlign': 'center'}),
    html.Hr(),

    html.P("Choisis le type de programme à exécuter :", style={'fontWeight': 'bold'}),
    dcc.RadioItems(
        id='mode-choice',
        options=[
            {'label': '🔹 Automatique (épaisseur + densité)', 'value': 'xrr_auto_dash.py'},
            {'label': '🔸 Manuel (sélection oscillations)', 'value': 'xrr_manual_dash.py'},
            {'label': '⚪ Couches fines (peu d’oscillations)', 'value': 'xrr_thinlayer_dash.py'},
        ],
        value='xrr_auto_dash.py',
        labelStyle={'display': 'block', 'marginBottom': '10px'}
    ),

    html.Button("🚀 Lancer l’analyse", id='launch-btn',
                style={'backgroundColor': '#007BFF', 'color': 'white', 'padding': '10px 20px'}),

    html.Div(id='output-msg', style={'marginTop': '30px', 'fontWeight': 'bold', 'whiteSpace': 'pre-wrap'})
])

# ===============================================================
# ⚙️ Callback : lancement du script choisi
# ===============================================================
@app.callback(
    Output('output-msg', 'children'),
    Input('launch-btn', 'n_clicks'),
    State('mode-choice', 'value')
)
def launch_program(n, script_name):
    if not n:
        return ""

    if not os.path.exists(script_name):
        return f"❌ Le script {script_name} est introuvable dans le dossier."

    # Ouvre le script choisi dans une nouvelle fenêtre Python
    try:
        subprocess.Popen([sys.executable, script_name])
        return f"✅ Le programme '{script_name}' a été lancé dans une nouvelle fenêtre.\n" \
               f"👉 Ouvre ton navigateur sur http://127.0.0.1:8050/"
    except Exception as e:
        return f"⚠️ Erreur au lancement : {e}"

# ===============================================================
# 🚀 Lancement
# ===============================================================
if __name__ == "__main__":
    app.run(debug=True, port=8060)

