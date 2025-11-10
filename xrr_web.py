# xrr_web.py
import os
import sys
import subprocess
import shlex
import platform
import base64
import dash
from dash import dcc, html, Input, Output, State

# ---------------------------
# CONFIG
# ---------------------------
SCRIPTS = {
    'auto': 'xrr_auto_dash.py',
    'manual': 'xrr_manual_dash.py',
    'thinlayer': 'xrr_thinlayer_dash.py'
}

# Map matériau to the string to inject when scripts ask input()
# We'll choose numbers consistent with your xrr_auto script (1..6).
MATERIAL_CHOICES = {
    'SiO2': '1',
    'Al2O3': '2',
    'TiO2_anatase': '3',
    'TiO2_rutile': '4',
    'Pt': '5',
    'Other': '6'
}

# ---------------------------
# DASH APP
# ---------------------------
app = dash.Dash(__name__)
app.title = "XRR Analysis - Launcher"

app.layout = html.Div([
    html.H1("XRR Analysis - Launcher"),
    html.P("1) Choisissez le mode :"),
    dcc.RadioItems(
        id='mode',
        options=[
            {'label': 'Automatique', 'value': 'auto'},
            {'label': 'Manuel', 'value': 'manual'},
            {'label': 'Fine couche', 'value': 'thinlayer'}
        ],
        value='auto',
        labelStyle={'display': 'inline-block', 'margin-right': '20px'}
    ),
    html.Hr(),
    html.P("2) Uploadez votre fichier .xy :"),
    dcc.Upload(
        id='upload',
        children=html.Div(['Glissez-déposez ou cliquez pour sélectionner un fichier (.xy)']),
        style={'width': '60%', 'height': '60px', 'lineHeight': '60px',
               'borderWidth': '1px', 'borderStyle': 'dashed',
               'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px auto'},
        multiple=False
    ),
    html.Div(id='file-info', style={'margin-top': '10px', 'color': '#006600'}),
    html.Hr(),
    html.P("3) Choisissez le matériau (pour la densité) :"),
    dcc.Dropdown(
        id='material',
        options=[
            {'label': 'SiO2 (amorphe)', 'value': 'SiO2'},
            {'label': 'Al2O3 (amorphe)', 'value': 'Al2O3'},
            {'label': 'TiO2 (anatase)', 'value': 'TiO2_anatase'},
            {'label': 'TiO2 (rutile)', 'value': 'TiO2_rutile'},
            {'label': 'Pt (platine)', 'value': 'Pt'},
            {'label': 'Autre', 'value': 'Other'}
        ],
        value='SiO2',
        style={'width': '40%'}
    ),
    html.Br(),
    html.Button("Lancer l'analyse (ouvre une console)", id='run', n_clicks=0,
                style={'padding': '10px 20px', 'font-size': '16px'}),
    html.Div(id='status', style={'margin-top': '20px', 'font-weight': 'bold'})
], style={'font-family': 'Arial', 'text-align': 'center', 'margin': '30px'})


# ---------------------------
# Helpers : save uploaded file and build launcher command
# ---------------------------
def save_uploaded_file(contents, filename):
    """Save uploaded base64 contents to cwd and return path."""
    if contents is None:
        return None
    header, data = contents.split(',', 1)
    binary = base64.b64decode(data)
    target = os.path.join(os.getcwd(), filename)
    with open(target, 'wb') as f:
        f.write(binary)
    return target


def build_exec_command(script_path, data_path, material_choice):
    """
    Build a Python -c command that:
      - sets file_path variable
      - defines builtins.input to return the material choice number (or name)
      - executes the script source (without modifying it on disk)
    This runs the script in a fresh interpreter.
    """
    # material answer that matches your scripts (they expect numbers 1..6)
    mat_answer = MATERIAL_CHOICES.get(material_choice, '6')  # default 'Other' -> '6'

    # Prepare python snippet. We'll:
    #  - import builtins
    #  - set file_path variable in globals()
    #  - monkeypatch builtins.input to return mat_answer (and for other inputs return '')
    #  - exec(open(script_path).read(), globals())
    # Use triple quotes to avoid quoting hell. We'll pass it as single -c string.
    py_snippet = f"""
import builtins, sys
# set file_path for the script
file_path = r'''{data_path}'''
# monkeypatch input() to return the material choice (first call)
_orig_input = builtins.input
def _fake_input(prompt=''):
    # return the choice number for the material first time,
    # then an empty string for subsequent calls (or keep returning last)
    return '{mat_answer}'
builtins.input = _fake_input
# execute the target script
code = open(r'{script_path}', 'r', encoding='utf-8').read()
exec(compile(code, r'{script_path}', 'exec'), globals())
# restore input (not strictly necessary in new process)
builtins.input = _orig_input
"""
    # On Windows, we'll pass the snippet as a single argument to python -c
    return py_snippet


def launch_in_new_process(py_code_snippet):
    """
    Launch a new Python interpreter that executes the given snippet.
    Use platform-specific flags to open a new console for interactivity (Windows).
    """
    python_exe = sys.executable

    # Write snippet to a temporary file and run that file in a new process.
    # This way we avoid quoting issues and allow editors to see script.
    import tempfile
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8')
    tmp.write(py_code_snippet)
    tmp.flush()
    tmp.close()
    cmd = [python_exe, tmp.name]

    # On Windows, create a new console window so matplotlib can open GUIs.
    if platform.system() == 'Windows':
        # CREATE_NEW_CONSOLE = 0x00000010
        creationflags = 0x00000010
        subprocess.Popen(cmd, creationflags=creationflags, cwd=os.getcwd())
    else:
        # On Unix (Linux/macOS), just spawn a new background process.
        # Note: GUI windows may not appear on headless servers.
        subprocess.Popen(cmd, cwd=os.getcwd())


# ---------------------------
# CALLBACKS
# ---------------------------
@app.callback(
    Output('file-info', 'children'),
    Input('upload', 'filename')
)
def show_filename(fn):
    if fn:
        return f"Fichier sélectionné: {fn}"
    return "Aucun fichier sélectionné."


@app.callback(
    Output('status', 'children'),
    Input('run', 'n_clicks'),
    State('mode', 'value'),
    State('upload', 'contents'),
    State('upload', 'filename'),
    State('material', 'value')
)
def run_script(nclicks, mode, contents, filename, material):
    if nclicks == 0:
        return ""
    if contents is None or filename is None:
        return "⚠️ Uploadez d'abord un fichier .xy."

    # Save uploaded file locally
    data_path = save_uploaded_file(contents, filename)
    if data_path is None:
        return "⚠️ Erreur lors de la sauvegarde du fichier."

    # Determine which script to run
    script_file = SCRIPTS.get(mode)
    if script_file is None:
        return "⚠️ Mode inconnu."

    script_path = os.path.join(os.getcwd(), script_file)
    if not os.path.exists(script_path):
        return f"⚠️ Script introuvable: {script_file} (placez-le dans le même dossier que xrr_web.py)."

    # Build execution snippet and launch in a new interpreter process
    snippet = build_exec_command(script_path, data_path, material)
    launch_in_new_process(snippet)

    return f"✅ Script {script_file} lancé en mode {mode} (matériau={material}). Une console s'ouvrira."

# ---------------------------
if __name__ == "__main__":
    # Use port env var if provided (good for deployments), else 8050
    port = int(os.environ.get("PORT", 8050))
    app.run_server(host="0.0.0.0", port=port, debug=True)
