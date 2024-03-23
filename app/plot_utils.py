import numpy as np
import plotly.graph_objects as go
from vars import *


custom_layout = {
    'plot_bgcolor': '#4e5567', # Couleur de fond du graphique
    'paper_bgcolor': '#2b323f', # Couleur de fond autour du graphique
    'font': {'color': 'white'}, # Couleur du texte
    'legend': {'bgcolor': '#2b323f'}, # Couleur de fond de la légende
    'xaxis': {
        'title_font': {'color': 'white'}, # Couleur du texte pour le titre de l'axe X
        'tickfont': {'color': 'white'}, # Couleur des marqueurs de l'axe X
        'gridcolor': 'darkgrey' # Couleur de la grille de l'axe X
    },
    'yaxis': {
        'title_font': {'color': 'white'}, # Couleur du texte pour le titre de l'axe Y
        'tickfont': {'color': 'white'}, # Couleur des marqueurs de l'axe Y
        'gridcolor': 'darkgrey' # Couleur de la grille de l'axe Y
    }
}

def calculate_stability(column):
    stability_df = dataprep.train.groupby(['date_mensuelle', column])['TARGET'].mean().unstack()
    #stability_df = df.groupby(['date_mensuelle', column])['TARGET'].mean().unstack()

    stability_df['stability'] = stability_df.std(axis=1) / stability_df.mean(axis=1)
    return stability_df

def plot_stability_plotly(variable):
    stability_df = calculate_stability(variable)

    # Création du graphique avec Plotly
    fig = go.Figure()

    # Ajout des tracés pour chaque classe
    for class_label in stability_df.drop('stability', axis=1).columns:
        values = stability_df[class_label]
        fig.add_trace(go.Scatter(x=stability_df.index, y=values, mode='lines+markers', name=f'Classe {class_label}'))

    # Mise en forme du graphique
    fig.update_layout(title=f'Stabilité de l\'impact sur la cible pour {variable}',
                      xaxis_title='Date',
                      yaxis_title='Proportion de la cible TARGET',
                      legend_title='Classes de_binned',
                      margin=dict(l=20, r=20, t=40, b=20))

    # Ajustement pour la légende à l'extérieur du graphique
    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))
    fig.update_layout(**custom_layout)

    return fig

def roc_curve():
    metrics = model.get_metrics()

    fpr = metrics["fpr"]
    tpr = metrics["tpr"]
    roc_auc = metrics["roc_auc"]

    #fpr = [0.03, 0.05, 0.1, 0.4, 0.6, 0.7]
    #tpr = [0.03, 0.1, 0.2, 0.5, 0.55, 0.8]
    #roc_auc = 0.7

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC curve (AUC = {:.2f})'.format(roc_auc),
                             line=dict(width=4)))

    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Chance', line=dict(width=2, dash='dash')))

    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        legend=dict(y=0.01, x=0.99, xanchor='right', yanchor='bottom'),
        margin=dict(l=40, r=0, t=40, b=30)
    )

    fig.update_layout(**custom_layout)

    return(fig)

def gini_coefficient(values):
    sorted_values = np.sort(values)
    n = len(values)
    cumulative_values_sum = np.cumsum(sorted_values)
    gini_index = (2 * np.sum(cumulative_values_sum) / (n * np.sum(sorted_values))) - (n + 1) / n
    return 1 - gini_index

def create_gini_figure():
    df = model.df_score.copy()
    if "date_trimestrielle" not in df.columns :
        df["date_mensuelle"] = pd.to_datetime(df["date_mensuelle"])
        df['date_trimestrielle'] = df['date_mensuelle'].dt.year.astype(str) + '_' + df['date_mensuelle'].dt.quarter.astype(str)

    fig = go.Figure()
    for classe in range(1, 8):
        df_classe = df[df['Classes'] == classe][["date_trimestrielle", "TARGET"]]
        grouped = df_classe.groupby(df_classe['date_trimestrielle'])["TARGET"]
        gini_per_year = grouped.apply(gini_coefficient)

        fig.add_trace(go.Scatter(x=gini_per_year.index, y=gini_per_year, mode='lines+markers',
                                 name=f'Classe {classe}'))

    fig.update_layout(
        title='Évolution Annuelle du Coefficient de Gini par Classe',
        xaxis_title='Année',
        yaxis_title='Coefficient de Gini',
        legend_title='Classe',
        template='plotly_white'
    )

    fig.update_layout(**custom_layout)

    return fig


def create_stability_figure():
    df = model.df_score.copy()
    if "date_trimestrielle" not in df.columns :
        df["date_mensuelle"] = pd.to_datetime(df["date_mensuelle"])
        df['date_trimestrielle'] = df['date_mensuelle'].dt.year.astype(str) + '_' + df['date_mensuelle'].dt.quarter.astype(str)

    fig = go.Figure()
    stability_df = df.groupby(['date_trimestrielle', 'Classes'])['TARGET'].mean().unstack()
    stability_df['stability'] = stability_df.std(axis=1) / stability_df.mean(axis=1)

    for class_label in stability_df.drop('stability', axis=1).columns:
        values = stability_df[class_label]
        fig.add_trace(go.Scatter(x=stability_df.index, y=values, mode='lines+markers',
                                 name=f'Classe {class_label}'))

    fig.update_layout(
        title=f'Stabilité de l\'impact sur la cible pour {"Classes"}',
        xaxis_title='Date',
        yaxis_title='Proportion de la cible TARGET',
        legend_title=f'Classes',
        template='plotly_white'
    )

    fig.update_layout(**custom_layout)

    return fig