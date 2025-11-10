import os
import sys
import numpy as np
import dash
from dash import dcc, html, Input, Output, State

# ajouter le dossier courant au path
script_dir = os.path.dirname(os.path.realpath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# importer tes 3 scripts existants
from xrr_auto_dash import run_auto
from xrr_manual_dash import run_manual
from xrr_thinlayer_dash import run_thinlayer

# ===============================================================
# Dash App
# ===============================================================
app = dash.Dash(__name__)
app.title = "XRR Analysis"

# ===============================================================
# Layout
# ===============================================================
app.layout = html.Div([
    html.H1("Bienvenue sur XRR Analyse"),
    html.Label("Choisissez le programme :"),
    dcc.Dropdown(
        id='program-dropdown',
        options=[
            {'label': 'Auto', 'value': 'auto'},
            {'label': 'Manuel', 'value': 'manual'},
            {'label': 'Thin Layer', 'value': 'thinlayer'}
        ],
        value='auto'
    ),
    html.Br(),
    html.Label("Upload du fichier XRR (.xy)"),
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Glissez-déposez ou sélectionnez un fichier']),
        style={'width': '100%', 'height': '60px', 'lineHeight': '60px',
               'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
               'textAlign': 'center', 'margin-bottom': '20px'},
        multiple=False
    ),
    html.Button('Lancer analyse', id='run-button', n_clicks=0),
    html.Br(),
    html.Div(id='results-div'),
    dcc.Graph(id='main-graph')
])

# ===============================================================
# Callback
# ===============================================================
@app.callback(
    Output('results-div', 'children'),
    Output('main-graph', 'figure'),
    Input('run-button', 'n_clicks'),
    State('program-dropdown', 'value'),
    State('upload-data', 'contents'),
)
def run_analysis(n_clicks, program, contents):
    if n_clicks == 0 or contents is None:
        return '', {}

    import io, base64
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    s = io.StringIO(decoded.decode('utf-8'))
    temp_path = "temp.xy"
    with open(temp_path, "w") as f:
        f.write(s.getvalue())

    # choisir le script selon le mode
    if program == "auto":
        t, rho_mass, two_theta, intensity, theta_peaks = run_auto(temp_path)
    elif program == "manual":
        t, rho_mass, two_theta, intensity, theta_peaks = run_manual(temp_path)
    else:
        t, rho_mass, two_theta, intensity, theta_peaks = run_thinlayer(temp_path)

    # graphique
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=two_theta, y=intensity, mode='lines', name='Signal'))
    fig.add_trace(go.Scatter(x=theta_peaks*2,
                             y=np.interp(theta_peaks*2, two_theta, intensity),
                             mode='markers', name='Pics',
                             marker=dict(color='red', size=10)))
    fig.update_layout(title='XRR Signal', xaxis_title='2θ (deg)', yaxis_title='Intensity', yaxis_type='log')

    results_text = f"Épaisseur : {t:.2f} nm | Densité : {rho_mass:.2f} g/cm³"
    return results_text, fig

# ===============================================================
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)

