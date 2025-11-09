import dash
from dash import html, dcc, Input, Output
import matplotlib.pyplot as plt
import io
import base64
import sys
import os

# Importer tes scripts existants
import xrr_auto_dash
import xrr_manual_dash
import xrr_thinlayer_dash

# ===============================================================
# Configuration Dash
# ===============================================================
app = dash.Dash(__name__)

app.layout = html.Div(
    style={'font-family': 'Arial', 'text-align': 'center', 'margin-top': '50px'},
    children=[
        html.H1("⚡ Analyse XRR - Interface Web", style={'color': '#003366'}),
        html.P("Choisissez le mode d'analyse :", style={'font-size': '18px'}),
        html.Div([
            html.Button("Mode automatique", id="btn-auto", n_clicks=0,
                        style={'margin': '10px', 'padding': '15px 25px', 'font-size': '16px'}),
            html.Button("Mode manuel", id="btn-manuel", n_clicks=0,
                        style={'margin': '10px', 'padding': '15px 25px', 'font-size': '16px'}),
            html.Button("Mode fine couche", id="btn-thin", n_clicks=0,
                        style={'margin': '10px', 'padding': '15px 25px', 'font-size': '16px'}),
        ]),
        html.Hr(),
        html.Div(id="output-text", style={'font-size': '18px', 'margin-top': '30px', 'color': '#006600'}),
        html.Img(id="output-graph")
    ]
)

# ===============================================================
# Fonction utilitaire : convertit figure Matplotlib en image PNG
# ===============================================================
def fig_to_uri(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    uri = base64.b64encode(buf.read()).decode('ascii')
    plt.close(fig)
    return "data:image/png;base64," + uri

# ===============================================================
# Callback pour exécuter les scripts originaux
# ===============================================================
@app.callback(
    Output("output-text", "children"),
    Output("output-graph", "src"),
    Input("btn-auto", "n_clicks"),
    Input("btn-manuel", "n_clicks"),
    Input("btn-thin", "n_clicks")
)
def launch_mode(n_auto, n_manual, n_thin):
    ctx = dash.callback_context
    if not ctx.triggered:
        return "Sélectionnez un mode pour lancer l'analyse.", None
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Chemin vers le fichier exemple (remplace-le par le vrai chemin)
    example_file = os.path.join(os.getcwd(), "EXEMPLE.xy")

    # Redirection stdout pour récupérer les prints
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        if button_id == "btn-auto":
            xrr_auto_dash.file_path = example_file
            with open("xrr_auto_dash.py", encoding="utf-8") as f:
                exec(f.read())

        elif button_id == "btn-manuel":
            xrr_manual_dash.file_path = example_file
            with open("xrr_manual_dash.py", encoding="utf-8") as f:
                exec(f.read())

        elif button_id == "btn-thin":
            xrr_thinlayer_dash.file_path = example_file
            with open("xrr_thinlayer_dash.py", encoding="utf-8") as f:
                exec(f.read())

        else:
            return "Erreur : mode inconnu", None

        # Récupère le texte print du script
        output_text = sys.stdout.getvalue()

        # Récupère la dernière figure Matplotlib
        fig = plt.gcf()
        img_uri = fig_to_uri(fig)

    finally:
        sys.stdout = old_stdout

    return output_text, img_uri

# ===============================================================
# Lancement du serveur
# ===============================================================
if __name__ == "__main__":
    app.run(debug=True, port=8050)

