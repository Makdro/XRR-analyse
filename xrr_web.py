import dash
from dash import html, dcc, Input, Output, State
import subprocess
import sys
import os

# ===============================================================
# 🌐 CONFIGURATION DE L'APPLICATION DASH
# ===============================================================
app = dash.Dash(__name__)
app.title = "Analyse XRR - Interface Web"

# Liste des matériaux pour la densité
materiaux = {
    'Al2O3': 3.95,  # exemple densité g/cm³
    'SiO2': 2.2,
    'AZ326MIF': 1.3
}

# ===============================================================
# 🖼️ LAYOUT DE L'APPLICATION
# ===============================================================
app.layout = html.Div(style={'font-family': 'Arial', 'text-align': 'center', 'margin': '30px'},
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
        html.Div(id='file-upload-div', style={'margin-top': '20px'}),
        html.Div(id="output", style={'font-size': '18px', 'margin-top': '30px', 'color': '#006600'})
    ]
)

# ===============================================================
# ⚙️ CALLBACK POUR AFFICHER LE UPLOAD APRÈS CHOIX DU MODE
# ===============================================================
@app.callback(
    Output('file-upload-div', 'children'),
    Input('btn-auto', 'n_clicks'),
    Input('btn-manuel', 'n_clicks'),
    Input('btn-thin', 'n_clicks'),
)
def show_upload(n_auto, n_manual, n_thin):
    ctx = dash.callback_context
    if not ctx.triggered:
        return ""
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    return html.Div([
        html.P("📂 Uploadez votre fichier XRR :", style={'font-size': '16px'}),
        dcc.Upload(
            id='upload-file',
            children=html.Div(['Glissez-déposez ou cliquez pour sélectionner le fichier']),
            style={
                'width': '50%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px auto'
            },
            multiple=False
        ),
        html.P("Choisissez le matériau pour la densité :"),
        dcc.Dropdown(
            id='dropdown-materiau',
            options=[{'label': k, 'value': k} for k in materiaux.keys()],
            value=list(materiaux.keys())[0],
            style={'width': '50%', 'margin': '0 auto'}
        ),
        html.Button("Lancer l'analyse", id='launch-btn', n_clicks=0,
                    style={'margin-top': '15px', 'padding': '10px 20px', 'font-size': '16px'})
    ])

# ===============================================================
# ⚙️ CALLBACK POUR LANCER LES PROGRAMMES
# ===============================================================
@app.callback(
    Output('output', 'children'),
    Input('launch-btn', 'n_clicks'),
    State('upload-file', 'contents'),
    State('upload-file', 'filename'),
    State('dropdown-materiau', 'value'),
    State('btn-auto', 'n_clicks'),
    State('btn-manuel', 'n_clicks'),
    State('btn-thin', 'n_clicks')
)
def launch_mode(n_launch, file_contents, filename, materiau, n_auto, n_manual, n_thin):
    if n_launch == 0 or file_contents is None:
        return "Sélectionnez un fichier et un mode pour lancer l'analyse."

    # Sauvegarde temporaire du fichier uploadé
    import base64
    import io
    data = file_contents.encode("utf8").split(b";base64,")[1]
    file_path = os.path.join(os.getcwd(), filename)
    with open(file_path, "wb") as f:
        f.write(base64.b64decode(data))

    ctx = dash.callback_context
    button_id = None
    # Détection du mode choisi
    if n_auto > 0:
        button_id = "btn-auto"
    elif n_manual > 0:
        button_id = "btn-manuel"
    elif n_thin > 0:
        button_id = "btn-thin"

    # Détermine quel script exécuter
    if button_id == "btn-auto":
        subprocess.Popen([sys.executable, "xrr_auto_dash.py", file_path, materiau],
                         creationflags=subprocess.CREATE_NEW_CONSOLE, cwd=os.getcwd())
        return f"🚀 Programme automatique lancé pour {materiau} !"

    elif button_id == "btn-manuel":
        subprocess.Popen([sys.executable, "xrr_manual_dash.py", file_path, materiau],
                         creationflags=subprocess.CREATE_NEW_CONSOLE, cwd=os.getcwd())
        return f"🧭 Programme manuel lancé pour {materiau} !"

    elif button_id == "btn-thin":
        subprocess.Popen([sys.executable, "xrr_thinlayer_dash.py", file_path, materiau],
                         creationflags=subprocess.CREATE_NEW_CONSOLE, cwd=os.getcwd())
        return f"🧪 Programme fine couche lancé pour {materiau} !"

    return "⚠️ Erreur : mode inconnu."

# ===============================================================
# 🚀 LANCEMENT DU SERVEUR DASH
# ===============================================================
if __name__ == "__main__":
    app.run(debug=True, port=8050)

