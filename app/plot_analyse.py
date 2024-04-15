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
    missing_percentages = missing_percentages[missing_percentages > 0]

    missing_percentages = missing_percentages.sort_values(ascending=False)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=missing_percentages.index,
        y=missing_percentages.values,  
        text=missing_percentages.values.round(2),
        textposition='auto',
        marker_color='lightsalmon'
    ))

    fig.update_layout(
        title='Pourcentage de valeurs manquantes par colonne',
        xaxis=dict(title='Colonnes'),
        yaxis=dict(title='Pourcentage de valeurs manquantes'),
        bargap=0.1,
        bargroupgap=0.1
    )

    if 'custom_layout' in globals():
        fig.update_layout(**custom_layout)

    return fig
def plot_correlation_matrix(target_variable):
    n_top = 10
    if target_variable not in analyse.df.columns:
        print(f"La colonne '{target_variable}' est absente.")
        return go.Figure()

    df_numeric = analyse.df.select_dtypes(include=[np.number])

    correlation_matrix = df_numeric.corr()

    target_correlations = correlation_matrix[target_variable].drop(target_variable)

    top_correlations = target_correlations.abs().sort_values(ascending=False).head(n_top)

    fig = go.Figure(go.Bar(
        x=top_correlations.index,
        y=top_correlations.values,
        marker_color=np.where(top_correlations.values >= 0, 'blue', 'red')
    ))

    fig.update_layout(
        title=f'Top {n_top} des variables les plus corrélées à {target_variable}',
        xaxis_title="Variables",
        yaxis_title="Coefficient de corrélation",
        xaxis_tickangle=-45
    )

    if 'custom_layout' in globals():
        fig.update_layout(**custom_layout)

    return fig

def plot_categorical_distribution(categorical_var):
    default = analyse.df[analyse.df[analyse.target] == 1][categorical_var]
    not_default = analyse.df[analyse.df[analyse.target] == 0][categorical_var]

    column_data_type = default.dtype

    if column_data_type == 'object':
        categories = sorted(set(default.unique()))
        fig = go.Figure()

        fig.add_trace(go.Histogram(x=default,
                                   nbinsx=len(categories),
                                   name='Défaut',
                                   opacity=0.7))

        fig.add_trace(go.Histogram(x=not_default,
                                   nbinsx=len(categories),
                                   name='Non Défaut',
                                   opacity=0.7))

        fig.update_layout(title=f'Distribution of {categorical_var}')
        fig.update_layout(**custom_layout)

    else:
        fig = None

    return fig


def plot_stability_plotly_analyse(variable):
    stability_df = analyse.df.groupby([analyse.date, variable])[analyse.target].mean().unstack()

    fig = go.Figure()

    for i, class_label in enumerate(stability_df.columns):
        values = stability_df[class_label]
        fig.add_trace(go.Scatter(x=stability_df.index,
                                 y=values,
                                 mode='lines+markers',
                                 name=f'Classe {class_label}'))

    fig.update_layout(title=f'Stabilité temporelle de {variable}',
                      xaxis_title='Date',
                      yaxis_title='Proportion de la cible',
                      legend_title='Classes',
                      margin=dict(l=20, r=20, t=40, b=20),
                      height=500)

    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.2,
        xanchor="center",
        x=0.5
    ))
    fig.update_layout(**custom_layout)

    return fig


def plot_stability_animated(variable):
    stability_taux_df = analyse.df.groupby(['date_mensuelle', variable])[analyse.target].mean().unstack()
    num_frames = 60
    step = len(stability_taux_df) // num_frames
    frames_data = [go.Frame(data=[go.Scatter(x=stability_taux_df.index[:i+1],
                                             y=stability_taux_df[col][:i+1],
                                             mode='lines', name=f'Classe {col}') for col in stability_taux_df.columns]) for i in range(0, len(stability_taux_df), step)]
    fig = go.Figure(
        data=[go.Scatter(x=stability_taux_df.index, y=stability_taux_df[col], mode='lines', name=f'Classe {col}') for col in stability_taux_df.columns],
        layout=go.Layout(
            title=f'Stabilité de taux pour {variable}',
            xaxis_title='Date',
            yaxis_title='Proportion de la cible TARGET',
            legend_title='Classes de_binned',
            xaxis_tickangle=45
        ),
        frames=frames_data
    )
    fig.update_layout(updatemenus=[dict(
        type="buttons",
        buttons=[dict(label="Play",
                      method="animate",
                      args=[None, dict(frame=dict(duration=30, redraw=False), fromcurrent=True, mode="immediate")])]
    )])
    fig.update_layout(**custom_layout)

    return fig
