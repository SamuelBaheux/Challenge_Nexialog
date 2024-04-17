import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from vars import *
import plotly.figure_factory as ff

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

def texte_analyse_globale():
    text = f"Informations globales sur les données : \n"
    text += f"- **Nombre de Lignes** : {analyse.df.shape[0]}\n"
    text += f"- **Nombre de Colonnes** : {analyse.df.shape[1]}\n"
    text += f"- **Taux de défaut** : {round(analyse.df[analyse.target].mean(), 2)*100}%\n"

    dtype_counts = analyse.df.dtypes.value_counts().to_dict()

    nbr_cat = dtype_counts[np.dtype("O")]
    nbr_num = dtype_counts[np.dtype('int64')] + dtype_counts[np.dtype('float64')]
    text += f"- **Répartition des Types de Données** :\n"
    text +=f"   - Catégorielles : {nbr_cat}\n"
    text +=f"   - Numériques : {nbr_num}\n"

    missing = analyse.df.isna().sum().sum()
    total_values = analyse.df.shape[0] * analyse.df.shape[1]
    missing_percent = (missing / total_values) * 100
    text += f"- **Valeurs Manquantes** : {missing_percent:.2f}%\n"

    duplicates = analyse.df.duplicated().sum()
    text += f"- **Données Dupliquées** : {duplicates}\n"
    return(text)

def texte_analyse_var(var):
    text = f"Informations sur la variable {var} :\n"

    if analyse.df[var].dtype == np.dtype("O") :
        type = 'Catégorielle'
    else :
        type = "Numérique"

    text+=f'- **Type de la variable** : {type}\n'
    text+=f'- **Nombre de valeurs uniques** : {analyse.df[var].nunique()}\n'

    missing = analyse.df[var].isna().sum()
    total_values = analyse.df.shape[0]
    missing_percent = (missing / total_values) * 100

    text += f"- **Valeurs Manquantes** : {missing_percent:.2f}%\n"

    desc = dictionnaire[dictionnaire['Row'] == var]['Description']
    if len(desc) > 0:
        text += f"- ** Description ** : {desc.values[0]}\n"

    return(text)



def plot_correlation_matrix(top):
    if analyse.target not in analyse.df.columns:
        print(f"La colonne '{analyse.target}' est absente.")
        return go.Figure()

    df_numeric = analyse.df.select_dtypes(include=[np.number])
    correlation_matrix = df_numeric.corr()
    target_correlations = correlation_matrix[analyse.target].drop(analyse.target)

    if top:
        top_vars = target_correlations.abs().sort_values(ascending=False).head(10).index
    else:
        top_vars = target_correlations.abs().sort_values(ascending=True).head(10).index

    top_vars = top_vars.insert(0, analyse.target)

    fig = go.Figure(go.Heatmap(
        x=top_vars,
        y=top_vars,
        z=correlation_matrix.loc[top_vars, top_vars].values,
        colorscale='RdBu',
        reversescale=True,
        zmid=0
    ))

    if top:
        fig.update_layout(
            title=f'Heatmap des 10 variables les plus corrélées à {analyse.target}',
            xaxis=dict(
                tickangle=-45,  # Inclinaison des étiquettes à -45 degrés
                tickfont=dict(size=10),  # Taille de police pour les ticks
            ),
        )
    else:
        fig.update_layout(
            title=f'Heatmap des 10 variables les moins corrélées à {analyse.target}',
        xaxis=dict(
            tickangle=-45,  # Inclinaison des étiquettes à -45 degrés
            tickfont=dict(size=10),  # Taille de police pour les ticks
        ),
        )

    fig.update_layout(width=750, height=600, **custom_layout)

    return fig


def missing_values():

    missing_percentages = analyse.df.isna().mean() * 100
    missing_percentages = missing_percentages[missing_percentages > 0]
    missing_percentages = missing_percentages.sort_values(ascending=False)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=missing_percentages.index,
        y=missing_percentages.values
    ))

    fig.update_layout(
        title='Pourcentage de valeurs manquantes par colonne',
        xaxis=dict(
            title='Colonnes',
            tickangle=-45,  # Inclinaison des étiquettes à -45 degrés
            tickfont=dict(size=10),  # Taille de police pour les ticks
        ),
        yaxis=dict(title='Pourcentage de valeurs manquantes'),
        bargap=0.1,
        bargroupgap=0.1,
        height = 500,
    )

    fig.update_layout(**custom_layout)

    return fig

def plot_stability_analyse(variable):
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

def plot_marginal_density(selected_column):
    df = analyse.df.copy()

    default = df[df[analyse.target] == 1][selected_column]
    not_default = df[df[analyse.target] == 0][selected_column]

    column_data_type = default.dtype
    print(column_data_type)

    if column_data_type in ['int64', 'float64']:
        fig = px.histogram(df,
                           x=selected_column,
                           title=f"Distribution de la variable {selected_column}")

        #fig.update_traces(fill='tozeroy')
        fig.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ))
        fig.update_layout(**custom_layout)

    else:
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

        fig.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ))

        fig.update_layout(title=f'Distribution conditionnelle au défaut de la variable {selected_column}')
        fig.update_layout(**custom_layout)

    return fig