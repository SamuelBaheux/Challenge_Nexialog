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
    # Calculer le pourcentage de valeurs manquantes pour chaque colonne
    missing_percentages = analyse.df.isna().mean() * 100

    # Filtrer pour conserver uniquement les colonnes avec des valeurs manquantes non nulles
    missing_percentages = missing_percentages[missing_percentages > 0]

    # Trier les pourcentages de valeurs manquantes
    missing_percentages = missing_percentages.sort_values(ascending=False)

    # Créer la figure
    fig = go.Figure()

    # Ajouter les données au graphique en tant que barres
    fig.add_trace(go.Bar(
        x=missing_percentages.index,  # Noms des colonnes
        y=missing_percentages.values,  # Pourcentages de valeurs manquantes
        text=missing_percentages.values.round(2),  # Texte à afficher au survol, arrondi à deux décimales
        textposition='auto',  # Position du texte sur les barres
        marker_color='lightsalmon'  # Couleur des barres
    ))

    # Mise en forme du graphique
    fig.update_layout(
        title='Pourcentage de valeurs manquantes par colonne',
        xaxis=dict(title='Colonnes'),
        yaxis=dict(title='Pourcentage de valeurs manquantes'),
        bargap=0.1,  # Espace entre les barres
        bargroupgap=0.1  # Espace entre les groupes de barres
    )

    # Appliquer la mise en page personnalisée
    if 'custom_layout' in globals():
        fig.update_layout(**custom_layout)

    return fig
def plot_correlation_matrix(target_variable):
    if target_variable not in analyse.df.columns:
            print(f"La colonne '{target_variable}' est absente.")
            return go.Figure()  # Retourner une figure vide si la variable cible n'est pas trouvée

    # S'assurer que toutes les colonnes sont numériques
    df_numeric = analyse.df.select_dtypes(include=[np.number])
    
    correlation_matrix = df_numeric.corr()

    # Créer un masque pour le triangle supérieur
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    # Appliquer le masque pour le triangle inférieur seulement
    tri_lower = correlation_matrix.where(~mask)
    
    # Mettre en valeur la colonne de la variable cible
    target_corr = tri_lower[target_variable].drop(target_variable)  # Retirer la corrélation de la target avec elle-même
    
    # Créer la figure avec une heatmap pour la matrice de corrélation en triangle
    fig = go.Figure(data=go.Heatmap(
        z=target_corr.values[None, :],  # Ajout d'une dimension pour compatibilité avec Heatmap
        x=target_corr.index,
        y=[target_variable],
        colorscale='Blues',
        zmin=-1,  # Définir les limites de l'échelle de couleur
        zmax=1
    ))

    # Mise en forme du graphique
    fig.update_layout(
        title=f'Corrélation des variables avec la variable cible {target_variable}',
        xaxis=dict(title='Variables'),
        yaxis=dict(title='Variable Cible', ticks='', side='left'),
        xaxis_tickangle=-45
    )

    if 'custom_layout' in globals():
        fig.update_layout(**custom_layout)

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
                      height = 500)

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
    frames_data = [go.Frame(data=[go.Scatter(x=stability_taux_df.index[:i+1], y=stability_taux_df[col][:i+1], mode='lines', name=f'Classe {col}') for col in stability_taux_df.columns]) for i in range(0, len(stability_taux_df), step)]
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

    return(fig)

def plot_marginal_density(var):
    colors = ['rgba(0, 0, 255, 0.5)', 'rgba(255, 165, 0, 0.5)']

    serie1 = analyse.df[analyse.df[analyse.target] == 1][var]
    serie2 = analyse.df[analyse.df[analyse.target] == 0][var]

    fig = go.Figure()

    fig.add_trace(go.Violin(x=serie1, line_color=colors[0], box_visible=False, width=1.5, side='positive'))
    fig.add_trace(go.Violin(x=serie2, line_color=colors[1], box_visible=False, width=1.5, side='positive'))

    fig.update_traces(orientation='h', points=False, width=1000)
    fig.update_layout(xaxis_showgrid=False, xaxis_zeroline=False, height=600)

    fig.update_layout(**custom_layout)

    return(fig)