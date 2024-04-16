import sys

from dash import dash

sys.path.append("./app")

from builders import *
from callbacks import *

app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
register_callbacks(app)

app.title = "NexiaMod"
server = app.server
app.config["suppress_callback_exceptions"] = True

app.layout = html.Div(
    id="big-app-container",
    children=[
        build_tabs(),
    ],
)

if __name__ == '__main__':
    app.run_server(debug=True)