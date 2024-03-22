import plotly.graph_objects as go
from vars import *

def calculate_stability(column):
    #stability_df = dataprep.train.groupby(['date_mensuelle', column])['TARGET'].mean().unstack()
    stability_df = dataprep.train.groupby(['date_mensuelle', column])['TARGET'].mean().unstack()

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

    return fig

def roc_curve():
    metrics = model.get_metrics()

    fpr = metrics["fpr"]
    tpr = metrics["tpr"]
    roc_auc = metrics["roc_auc"]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC curve (AUC = {:.2f})'.format(roc_auc),
                             line=dict(color='darkorange', width=2)))

    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Chance', line=dict(color='navy', width=2, dash='dash')))

    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        legend=dict(y=0.01, x=0.99, xanchor='right', yanchor='bottom'),
        margin=dict(l=40, r=0, t=40, b=30)
    )

    return(fig)