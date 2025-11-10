import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LinearRegression
import dash
from dash import dcc, html, Input, Output, State
import base64
import io
import plotly.graph_objects as go

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

    html.Button("Lancer l'analyse", id="run-button", n_clicks=0),

    html.Br(),
    html.Label("Choisissez le programme d'analyse"),
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
        style={
            'width': '100%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
            'textAlign': 'center', 'margin-bottom': '20px'
        },
        multiple=False
    ),

    html.Br(),
    html.Label("Choisissez le matériau pour calcul densité"),
    dcc.Dropdown(
        id='material-dropdown',
        options=[
            {'label': 'SiO2 (amorphe)', 'value': 'SiO2'},
            {'label': 'Al2O3 (amorphe)', 'value': 'Al2O3'},
            {'label': 'TiO2 (anatase)', 'value': 'TiO2_anatase'},
            {'label': 'TiO2 (rutile)', 'value': 'TiO2_rutile'},
            {'label': 'Pt (platine)', 'value': 'Pt'},
            {'label': 'Autre', 'value': 'Other'}
        ],
        value='SiO2'
    ),

    html.Br(),
    html.Button('Lancer analyse', id='run-button', n_clicks=0),
    html.Br(),
    html.Div(id='results-div'),
    dcc.Graph(id='main-graph')
])

# ===============================================================
# Matériaux pour densité
# ===============================================================
materiaux_info = {
    "SiO2": {"M": 60.08, "Z": 30, "dens": 2.20},
    "Al2O3": {"M": 101.96, "Z": 50, "dens": 3.97},
    "TiO2_anatase": {"M": 79.87, "Z": 38, "dens": 3.90},
    "TiO2_rutile": {"M": 79.87, "Z": 38, "dens": 4.23},
    "Pt": {"M": 195.08, "Z": 78, "dens": 21.45}
}

# ===============================================================
# Helper functions
# ===============================================================
def parse_contents(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    s = io.StringIO(decoded.decode('utf-8'))
    data = np.loadtxt(s)
    two_theta = data[:, 0]
    intensity = data[:, 1]
    return two_theta, intensity

# ===============================================================
# Callback
# ===============================================================
@app.callback(
    Output('results-div', 'children'),
    Output('main-graph', 'figure'),
    Input('run-button', 'n_clicks'),
    State('program-dropdown', 'value'),
    State('upload-data', 'contents'),
    State('material-dropdown', 'value')
)
def run_analysis(n_clicks, program, contents, material_choice):
    if n_clicks == 0 or contents is None:
        return '', {}

    # ===========================================================
    # Constants
    lambda_Cu = 0.15418  # nm
    r_e = 2.8179403262e-5  # Å
    N_A = 6.02214076e23

    # ===========================================================
    # Parse file
    two_theta, intensity = parse_contents(contents)

    # ===========================================================
    # Densité et matériau
    if material_choice in materiaux_info:
        mat = materiaux_info[material_choice]
        M, Z = mat['M'], mat['Z']
    else:
        M = 60.0  # valeur par défaut
        Z = 30

    # ===========================================================
    # Auto mode (détection automatique)
    if program == 'auto':
        sigma_smooth = 5
        intensity_smooth = gaussian_filter1d(intensity, sigma=sigma_smooth)
        intensity_detrend = intensity / intensity_smooth
        signal_log = np.log10(intensity + 1)
        second_derivative = np.gradient(np.gradient(signal_log))
        peaks, _ = find_peaks(second_derivative, prominence=0.01)
        peaks_selected = peaks[:10]
        theta_peaks = two_theta[peaks_selected] / 2

    # Manual et Thinlayer (simplifié ici)
    else:
        intensity_smooth = gaussian_filter1d(intensity, sigma=5)
        peaks, _ = find_peaks(intensity_smooth, distance=5)
        theta_peaks = two_theta[peaks[:10]] / 2

    # ===========================================================
    # Fit linéaire pour épaisseur
    m = np.arange(1, len(theta_peaks) + 1)
    m2 = m ** 2
    theta2 = np.deg2rad(theta_peaks) ** 2
    reg = LinearRegression().fit(m2.reshape(-1, 1), theta2)
    a = reg.coef_[0]
    t = np.sqrt(lambda_Cu ** 2 / (4 * a))

    # ===========================================================
    # Densité
    theta_c_rad = np.deg2rad(theta_peaks[0])
    rho_e = (np.pi * theta_c_rad ** 2) / (r_e * lambda_Cu ** 2)
    rho_mass = rho_e * M / (Z * N_A) * 1e24

    # ===========================================================
    # Graphique
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=two_theta, y=intensity, mode='lines', name='Signal'))
    fig.add_trace(go.Scatter(
        x=theta_peaks * 2,
        y=np.interp(theta_peaks * 2, two_theta, intensity),
        mode='markers', name='Pics', marker=dict(color='red', size=10)
    ))
    fig.update_layout(
        title='XRR Signal',
        xaxis_title='2θ (deg)',
        yaxis_title='Intensity',
        yaxis_type='log'
    )

    # ===========================================================
    results_text = f"Épaisseur : {t:.2f} nm | Densité : {rho_mass:.2f} g/cm³"
    return results_text, fig


# ===============================================================
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
