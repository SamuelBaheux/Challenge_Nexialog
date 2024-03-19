from dash.dependencies import Input, Output, State
import dash
from dash import html

INFO_VAR_INT = '''Détail des variables choisies pour l\'interprétabilité :
- Variable A : zhfrhg
- Variable B : gkrkg'''


INFO_VAR_PERF =  '''Détail des variables choisies pour la performance :
- Variable A : zhfrhg
- Variable B : gkrkg'''


def register_callbacks(app):
    @app.callback(
        [Output('result-output', 'children'),
         Output('app-tabs', 'value')],  # Change l'onglet actif en plus de mettre à jour le contenu
        [Input('launch-button', 'n_clicks')],
        prevent_initial_call=True
    )
    def update_result(n_clicks):
        if n_clicks > 0:
            # Ici, adaptez la logique pour générer les résultats spécifiques à votre application
            results = html.Div([
                html.H3('Résultats'),
                # Incluez ici les éléments HTML pour afficher les résultats
            ])
            return results, 'tab2'  # Change vers l'onglet des résultats

    @app.callback(
        [Output('variables-dropdown', 'value'),
         Output('variables-info-markdown', 'children')],
        [Input('interpretabilite-button', 'n_clicks'),
         Input('performance-button', 'n_clicks')],
        [State('variables-dropdown', 'value')]
    )
    def update_dropdown_variables(interpretabilite_n_clicks, performance_n_clicks, existing_values):
        ctx = dash.callback_context
        info_text = ''

        if not ctx.triggered:
            button_id = 'No clicks yet'
        else:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if button_id == 'interpretabilite-button':
            new_values = ['A', 'B']  # Remplacer par les vraies variables pour l'interprétabilité
            info_text = INFO_VAR_INT
        elif button_id == 'performance-button':
            new_values = ['B', 'C']  # Remplacer par les vraies variables pour la performance
            info_text =INFO_VAR_PERF
        else:
            new_values = existing_values

        return new_values, info_text
