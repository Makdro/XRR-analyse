import base64
import io
import numpy as np
import dash
from dash import html, dcc, Input, Output, State
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_filter1d

# ===============================================================
# ⚙️ Paramètres de base
# ===============================================================
lambda_Cu = 0.15418  # nm (Cu Kα)
theta_min = 0.2
theta_max = 4.0
sigma_smooth = 5

# ===============================================================
# 🚀 Application Dash
# ===============================================================
app = dash.Dash(__name__)
app.title = "XRR Manuel – Voltify"

app.layout = html.Div([
    html.H2("XRR Manuel – Sélection interactive des oscillations", style={'textAlign': 'center'}),
    html.Hr(),

    html.Label("1️⃣ Importer un fichier .xy :", style={'fontWeight': 'bold'}),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            '📂 Glissez-déposez un fichier ici ou ',
            html.A('cliquez pour sélectionner')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'marginBottom': '20px'
        },
        multiple=False
    ),

    html.Div(id='file-info', style={'marginBottom': '20px'}),

    html.Label("2️⃣ Sélectionnez les pics sur le graphe :", style={'fontWeight': 'bold'}),
    html.P("(Cliquez sur les oscillations visibles dans la zone utile)"),
    dcc.Graph(id='graph-xrr', style={'height': '500px'}),

    html.Div(id='selected-points', style={'margin': '10px 0', 'fontWeight': 'bold'}),

    html.Button("3️⃣ Calculer l'épaisseur", id='compute-btn',
                style={'backgroundColor': '#007BFF', 'color': 'white', 'padding': '10px 20px'}),
    html.Div(id='results', style={'marginTop': '20px', 'whiteSpace': 'pre-wrap'})
])

# ===============================================================
# 🧩 Fonctions internes
# ===============================================================
def parse_xy(contents):
    """Décodage du fichier .xy uploadé."""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        data = np.loadtxt(io.StringIO(decoded.decode('utf-8')))
        two_theta = data[:, 0]
        intensity = data[:, 1]
        return two_theta, intensity
    except Exception as e:
        raise ValueError(f"Erreur de lecture du fichier : {e}")

# ===============================================================
# 📈 Callback : affichage du signal XRR
# ===============================================================
@app.callback(
    Output('graph-xrr', 'figure'),
    Output('file-info', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_graph(contents, filename):
    if contents is None:
        return go.Figure(), "Aucun fichier chargé."

    two_theta, intensity = parse_xy(contents)
    mask = (two_theta >= theta_min) & (two_theta <= theta_max)
    two_theta_zone = two_theta[mask]
    intensity_zone = intensity[mask]
    intensity_smooth = gaussian_filter1d(intensity_zone, sigma=sigma_smooth)
    intensity_detrend = intensity_zone / intensity_smooth

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=two_theta_zone, y=intensity_detrend,
        mode='lines', name='Signal détendré', line=dict(color='blue')
    ))
    fig.update_layout(
        title="Signal XRR détendré (cliquez sur les oscillations)",
        xaxis_title="2θ (°)",
        yaxis_title="Signal détendré",
        hovermode="x unified"
    )
    return fig, f"✅ Fichier chargé : {filename}"

# ===============================================================
# 📍 Callback : affichage des points sélectionnés
# ===============================================================
@app.callback(
    Output('selected-points', 'children'),
    Input('graph-xrr', 'clickData'),
    State('selected-points', 'children')
)
def select_peaks(clickData, current_text):
    if clickData is None:
        return "Aucun pic sélectionné pour l'instant."
    x_val = clickData['points'][0]['x']
    if current_text and "Sélectionnés" in current_text:
        try:
            existing = [float(v) for v in current_text.split(":")[1].split(",")]
        except Exception:
            existing = []
    else:
        existing = []
    existing.append(x_val)
    existing = sorted(set(existing))
    return f"✅ Pics sélectionnés ({len(existing)}) : {', '.join(f'{x:.3f}' for x in existing)}"

# ===============================================================
# 🧮 Callback : calcul de l'épaisseur
# ===============================================================
@app.callback(
    Output('results', 'children'),
    Input('compute-btn', 'n_clicks'),
    State('selected-points', 'children')
)
def compute_thickness(n, selected_text):
    if not n:
        return ""
    if not selected_text or "Sélectionnés" not in selected_text:
        return "⚠️ Aucun pic sélectionné !"

    # Récupérer les angles choisis
    theta_peaks = [float(v) for v in selected_text.split(":")[1].split(",")]
    if len(theta_peaks) < 2:
        return "⚠️ Il faut au moins deux oscillations pour calculer l'épaisseur."

    theta_peaks = np.array(sorted(theta_peaks))
    theta_rad = np.deg2rad(theta_peaks / 2)
    m = np.arange(1, len(theta_rad) + 1)
    m2 = m**2
    theta2 = theta_rad**2

    reg = LinearRegression().fit(m2.reshape(-1, 1), theta2)
    a = reg.coef_[0]
    b = reg.intercept_
    r2 = reg.score(m2.reshape(-1,1), theta2)
    t = np.sqrt(lambda_Cu**2 / (4 * a))

    result_txt = (
        "==============================\n"
        f"Épaisseur estimée : {t:.2f} nm\n"
        f"Coefficient a = {a:.3e}\n"
        f"R² = {r2:.4f}\n"
        "=============================="
    )
    return result_txt

# ===============================================================
# 🚀 Lancement
# ===============================================================
if __name__ == '__main__':
    app.run(debug=True, port=8050)


