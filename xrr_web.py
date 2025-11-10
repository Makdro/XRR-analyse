import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import numpy as np
import io
import base64

# Import des 3 scripts
import xrr_auto_dash
import xrr_manual_dash
import xrr_thinlayer_dash

# ===============================================================
# INITIALISATION DE L’APP DASH
# ===============================================================
app = dash.Dash(__name__)
server = app.server  # nécessaire pour Render

# ===============================================================
# MISE EN PAGE
# ===============================================================
app.layout = html.Div([
    html.H1("Analyse XRR – Interface web", style={"textAlign": "center"}),

    html.Hr(),

    # Sélection du mode
    html.Div([
        html.Label("🧭 Choisir le mode d'analyse :"),
        dcc.Dropdown(
            id="mode",
            options=[
                {"label": "Automatique", "value": "auto"},
                {"label": "Manuel", "value": "manual"},
                {"label": "Thin Layer", "value": "thinlayer"},
            ],
            value="auto",
            clearable=False,
            style={"width": "50%"}
        )
    ], style={"textAlign": "center", "marginBottom": "20px"}),

    # Téléversement de fichier
    html.Div([
        html.Label("📂 Importer un fichier XRR (.xy) :"),
        dcc.Upload(
            id="upload-data",
            children=html.Div(["Glisser-déposer ou ", html.A("sélectionner un fichier")]),
            style={
                "width": "80%", "height": "80px", "lineHeight": "80px",
                "borderWidth": "2px", "borderStyle": "dashed",
                "borderRadius": "10px", "textAlign": "center", "margin": "auto"
            },
            multiple=False
        )
    ], style={"textAlign": "center"}),

    html.Br(),
    html.Div(id="output-message", style={"textAlign": "center", "color": "red"}),

    html.Hr(),

    # Graphique
    dcc.Graph(id="xrr-graph", style={"height": "600px"}),

    # Résultats
    html.Div(id="results", style={"textAlign": "center", "fontSize": "20px", "marginTop": "30px"})
])

# ===============================================================
# CALLBACK PRINCIPAL
# ===============================================================
@app.callback(
    [Output("xrr-graph", "figure"),
     Output("results", "children"),
     Output("output-message", "children")],
    [Input("upload-data", "contents"),
     Input("mode", "value")],
    [State("upload-data", "filename")]
)
def update_output(contents, mode, filename):
    if contents is None:
        return go.Figure(), "", "⚠️ Veuillez importer un fichier .xy"

    try:
        # Lecture du fichier uploadé
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        data = np.loadtxt(io.BytesIO(decoded))

        two_theta = data[:, 0]
        intensity = data[:, 1]

        # Exécution selon le mode choisi
        if mode == "auto":
            t, rho_mass, peaks = xrr_auto_dash.run_auto(two_theta, intensity)
            result_text = f"🧠 Mode Automatique → Épaisseur = {t:.2f} nm | ρ = {rho_mass:.2f} g/cm³"

        elif mode == "manual":
            t, peaks = xrr_manual_dash.run_manual(two_theta, intensity)
            result_text = f"🎯 Mode Manuel → Épaisseur = {t:.2f} nm"

        elif mode == "thinlayer":
            t, rho_mass = xrr_thinlayer_dash.run_thinlayer(two_theta, intensity)
            result_text = f"💧 Thin Layer → Épaisseur = {t:.2f} nm | ρ = {rho_mass:.2f} g/cm³"

        # Création du graphique
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=two_theta, y=intensity, mode="lines", name="Signal XRR"))
        if 'peaks' in locals():
            fig.add_trace(go.Scatter(
                x=peaks,
                y=np.interp(peaks, two_theta, intensity),
                mode="markers",
                name="Pics sélectionnés",
                marker=dict(color="red", size=8)
            ))
        fig.update_layout(
            xaxis_title="2θ (deg)",
            yaxis_title="Intensité (a.u.)",
            yaxis_type="log",
            template="plotly_white"
        )

        return fig, result_text, ""

    except Exception as e:
        return go.Figure(), "", f"❌ Erreur : {e}"

# ===============================================================
# LANCEMENT LOCAL
# ===============================================================
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)
