import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from vars import *

custom_layout = {
    'plot_bgcolor': '#4e5567',
    'paper_bgcolor': '#2b323f',
    'font': {'color': 'white'},
    'legend': {'bgcolor': '#2b323f'},
    'xaxis': {
        'title_font': {'color': 'white'},
        'tickfont': {'color': 'white'},
        'gridcolor': 'darkgrey'
    },
    'yaxis': {
        'title_font': {'color': 'white'},
        'tickfont': {'color': 'white'},
        'gridcolor': 'darkgrey'
    }
}



def missing_values():

    missing_percentages = analyse.df.isna().mean() * 100
    print(missing_percentages)
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=missing_percentages.index,  # Noms des colonnes
        y=missing_percentages.values,  # Pourcentages de valeurs manquantes
        text=missing_percentages.values.round(2),  # Texte à afficher au survol
        textposition='auto',  # Position du texte
        marker_color='lightsalmon'  # Couleur des barres
    ))

    # Mise en forme de la figure
    fig.update_layout(
        title='Pourcentage de valeurs manquantes par colonne',
        xaxis=dict(title='Colonnes'),
        yaxis=dict(title='Pourcentage de valeurs manquantes'),
        bargap=0.1,  # Espace entre les barres
        bargroupgap=0.1  # Espace entre les groupes de barres
    )

    fig.update_layout(**custom_layout)

    return fig