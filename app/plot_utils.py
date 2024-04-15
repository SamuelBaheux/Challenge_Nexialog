import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

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


def plot_stability_plotly(variable):
    stability_df = dataprep.train.groupby([dataprep.date, variable])[dataprep.target].mean().unstack()

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

def plot_hist(column):
    histogramme = go.Figure(go.Histogram(x=dataprep.train[column]))

    histogramme.update_layout(
        title=f'Distribution de {column}',
        xaxis_title=column,
        yaxis_title='Fréquence',
        bargap=0.2,
        height=520
    )

    histogramme.update_layout(**custom_layout)

    return histogramme

def courbe_roc(model):
    metrics = model.get_metrics()
    fpr = metrics["fpr"]
    tpr = metrics["tpr"]
    roc_auc = metrics["roc_auc"]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=fpr,
                             y=tpr,
                             mode='lines',
                             name='Courbe ROC (AUC = {:.2f})'.format(roc_auc),
                             line=dict(width=4)))

    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                             mode='lines',
                             name='Chance',
                             line=dict(width=2, dash='dash')))

    fig.update_layout(title='Courbe ROC',
                      xaxis_title='Taux Faux Positifs',
                      yaxis_title='Taux Vrais Positifs',
                      legend=dict(y=0.01, x=0.99, xanchor='right', yanchor='bottom'),
                      margin=dict(l=40, r=0, t=40, b=30))

    fig.update_layout(**custom_layout)

    return(fig)

def gini_coefficient(values):
    sorted_values = np.sort(values)
    n = len(values)
    cumulative_values_sum = np.cumsum(sorted_values)
    gini_index = (2 * np.sum(cumulative_values_sum) / (n * np.sum(sorted_values))) - (n + 1) / n
    return 1 - gini_index

def create_gini_figure(model):
    df = model.df_score.copy()
    if "date_trimestrielle" not in df.columns :
        df[dataprep.date] = pd.to_datetime(df[dataprep.date])
        df['date_trimestrielle'] = df[dataprep.date].dt.year.astype(str) + '_' + df[dataprep.date].dt.quarter.astype(str)

    fig = go.Figure()
    for classe in range(1, 8):
        df_classe = df[df['Classes'] == classe][["date_trimestrielle", dataprep.target]]
        grouped = df_classe.groupby(df_classe['date_trimestrielle'])[dataprep.target]
        gini_per_year = grouped.apply(gini_coefficient)

        fig.add_trace(go.Scatter(x=gini_per_year.index, y=gini_per_year, mode='lines+markers',
                                 name=f'Classe {classe}'))

    fig.update_layout(title='Évolution Temporelle du Coefficient de Gini par Classe Homogène de Risque',
                      xaxis_title='Date',
                      yaxis_title='Coefficient de Gini',
                      legend_title='Classe',
                      template='plotly_white')

    fig.update_layout(xaxis=dict(tickangle=-45))

    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))

    fig.update_layout(**custom_layout)

    return fig



def create_stability_figure(model):
    df = model.df_score.copy()
    if "date_trimestrielle" not in df.columns :
        df[dataprep.date] = pd.to_datetime(df[dataprep.date])
        df['date_trimestrielle'] = df[dataprep.date].dt.year.astype(str) + '_' + df[dataprep.date].dt.quarter.astype(str)

    fig = go.Figure()
    stability_df = df.groupby(['date_trimestrielle', 'Classes'])[dataprep.target].mean().unstack()

    for class_label in stability_df.columns:
        values = stability_df[class_label]
        fig.add_trace(go.Scatter(x=stability_df.index,
                                 y=values,
                                 mode='lines+markers',
                                 name=f'Classe {class_label}'))

    fig.update_layout(title=f'Stabilité temporelle des Classes Homogènes de Risque',
                      xaxis_title='Date',
                      yaxis_title='Proportion de la cible',
                      legend_title=f'Classes',
                      template='plotly_white')

    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))

    fig.update_layout(xaxis=dict(tickangle=-45))
    fig.update_layout(**custom_layout)

    return fig

def plot_shap_values():
    if model_challenger.model_name == "xgb" :
        shap_values = model_challenger.model.shap_values
        train = model_challenger.model.X_train

        shap_values = pd.DataFrame(shap_values, columns=train.columns)
        train.reset_index(inplace=True, drop=True)

        replacements = {
            'zz': '[',
            'vv': ']',
            'ww': ';',
            'ff': '-',
            'pp': '.'
        }

        for old, new in replacements.items():
            train.columns = [col.replace(old, new) for col in train.columns]
            shap_values.columns = [col.replace(old, new) for col in shap_values.columns]

        train = train.iloc[:500, :]
        shap_values = shap_values.iloc[:500, :]

        # Joining SHAP values and one-hot encoded features
        merged_df = shap_values.join(train, lsuffix='_shap', rsuffix='_train')

        # Melt the merged DataFrame to long format
        melted_df = merged_df.melt(value_vars=[col for col in
                                               merged_df.columns if '_shap' in col],
                                   var_name='Feature',
                                   value_name='SHAP Value')

        melted_df['Feature'] = melted_df['Feature'].str.replace('_shap', '')

        for feature in train.columns:
            feature_shap = feature + '_shap'
            feature_train = feature + '_train'
            melted_df.loc[melted_df['Feature'] == feature, 'One-hot Value'] = merged_df[feature_train].values

        # Generate the plot again
        fig = px.strip(melted_df, x='SHAP Value', y='Feature',
                       color='One-hot Value',
                       orientation='h', stripmode='overlay',
                       title='Impact des Variables sur la sortie du modèle')

        fig.update_layout(xaxis_title='Valeurs de Shapley',
                          yaxis_title='Variable')

        fig.update_layout(**custom_layout)

        return(fig)


def plot_metrics_leftpanel(metrics) :
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=[metrics],
        y=[metrics],
        text=[f'{round(metrics)}%'],
        textposition='auto',
        orientation='h',
    ))

    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 100]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        height=40,

    )
    fig.update_layout(**custom_layout)

    fig.update_layout(paper_bgcolor = '#4e5567')

    return(fig)

def update_graph_dist_column(selected_column, model):
    default = model.df_score[model.df_score[dataprep.target] == 1][selected_column]
    not_default = model.df_score[model.df_score[dataprep.target] == 0][selected_column]

    column_data_type = default.dtype

    if column_data_type in ['int64', 'float64']:

        fig = ff.create_distplot(hist_data=[default, not_default],
                                 group_labels=['Défaut', 'Non Défaut'],
                                 bin_size=0.2,
                                 show_rug=False,
                                 show_hist=False)

        fig.update_layout(title=f'Distribution conditionnelle au défaut de la variable {selected_column}')
        fig.update_traces(fill='tozeroy')
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



