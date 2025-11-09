import dash
from dash import dcc, html, Input, Output, State
import base64
import io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import gaussian_filter1d
import plotly.graph_objs as go
import tempfile

# ===============================================================
# 🌐 INIT DASH
# ===============================================================
app = dash.Dash(__name__)
server = app.server  # nécessaire pour Render

# ===============================================================
# 📄 LAYOUT
# ===============================================================
app.layout = html.Div([
    html.H1("⚡ Analyse XRR - Interface Web"),
    html.P("Choisissez le mode d'analyse :"),
    
    html.Div([
        html.Button("Mode automatique", id="btn-auto", n_clicks=0, style={'margin': '5px'}),
        html.Button("Mode manuel", id="btn-manuel", n_clicks=0, style={'margin': '5px'}),
        html.Button("Mode fine couche", id="btn-thin", n_clicks=0, style={'margin': '5px'}),
    ]),

    html.Hr(),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Glissez-déposez ou ',
            html.A('sélectionnez un fichier XRR')
        ]),
        style={
            'width': '50%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),
    html.Div(id='output-results', style={'margin-top': '20px'})
])

# ===============================================================
# 🔧 FONCTIONS DE TRAITEMENT XRR
# ===============================================================
def parse_contents(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    # On retourne un np.array
    data = np.loadtxt(io.StringIO(decoded.decode('utf-8')))
    return data[:,0], data[:,1]

def run_auto(two_theta, intensity):
    # Lissage
    intensity_smooth = gaussian_filter1d(intensity, sigma=5)
    intensity_detrend = intensity / intensity_smooth

    # Détection de pics (simplifiée)
    peaks, _ = find_peaks(intensity_detrend, distance=5)
    theta_peaks = two_theta[peaks][:10]  # on prend max 10 pics

    # Fit linéaire
    theta_rad = np.deg2rad(theta_peaks/2)
    m = np.arange(1,len(theta_rad)+1)
    m2 = m**2
    theta2 = theta_rad**2
    reg = LinearRegression().fit(m2.reshape(-1,1), theta2)
    a = reg.coef_[0]
    t = np.sqrt(0.15418**2 / (4*a))  # Cu Kα

    # Graphique Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=two_theta, y=intensity_detrend, mode='lines', name='Signal détendré'))
    fig.add_trace(go.Scatter(x=theta_peaks, y=np.interp(theta_peaks,two_theta,intensity_detrend),
                             mode='markers', name='Pics détectés'))
    fig.update_layout(title=f"Mode automatique - épaisseur ≈ {t:.2f} nm",
                      xaxis_title='2θ (deg)',
                      yaxis_title='Signal détendré')
    return fig, f"Épaisseur estimée : {t:.2f} nm"

def run_manual(two_theta, intensity):
    # Simplification : juste montrer signal détendré
    intensity_smooth = gaussian_filter1d(intensity, sigma=5)
    intensity_detrend = intensity / intensity_smooth
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=two_theta, y=intensity_detrend, mode='lines', name='Signal détendré'))
    fig.update_layout(title="Mode manuel - sélection des pics possible",
                      xaxis_title='2θ (deg)',
                      yaxis_title='Signal détendré')
    return fig, "Sélectionnez manuellement les pics sur votre fichier XRR localement."

def run_thin(two_theta, intensity):
    # Simplification pour fine couche
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=two_theta, y=intensity, mode='lines', name='Signal brut'))
    fig.update_layout(title="Mode fine couche - signal brut",
                      xaxis_title='2θ (deg)',
                      yaxis_title='Intensité (a.u.) [log]")
    return fig, "Analyse fine couche - pics très larges, traitement manuel conseillé."

# ===============================================================
# ⚙️ CALLBACK
# ===============================================================
@app.callback(
    Output('output-results','children'),
    Input('btn-auto','n_clicks'),
    Input('btn-manuel','n_clicks'),
    Input('btn-thin','n_clicks'),
    State('upload-data','contents')
)
def launch_mode(n_auto, n_manual, n_thin, contents):
    if contents is None:
        return "⚠️ Veuillez uploader un fichier XRR pour lancer l'analyse."

    two_theta, intensity = parse_contents(contents)

    ctx = dash.callback_context
    if not ctx.triggered:
        return "Sélectionnez un mode pour lancer une analyse."
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == "btn-auto":
        fig, text = run_auto(two_theta,intensity)
    elif button_id == "btn-manuel":
        fig, text = run_manual(two_theta,intensity)
    elif button_id == "btn-thin":
        fig, text = run_thin(two_theta,intensity)
    else:
        return "Erreur : mode inconnu."

    return html.Div([
        html.P(text, style={'font-weight':'bold'}),
        dcc.Graph(figure=fig)
    ])

# ===============================================================
# 🚀 RUN SERVER
# ===============================================================
if __name__=="__main__":
    import os
    port = int(os.environ.get("PORT", 8050))
    app.run_server(host="0.0.0.0", port=port)
