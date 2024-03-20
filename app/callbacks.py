from dash.dependencies import Input, Output, State
import dash
from dash import html

INFO_VAR_INT = '''Détail des variables choisies pour l\'interprétabilité :
- REGION_RATING_CLIENT_W_CITY : Rating of the region where client lives with taking city into account
- DAYS_CREDIT_ENDDATE : Remaining duration of CB credit (in days) at the time of application in Home Credit
- RATE_DOWN_PAYMENT : Down payment rate normalized on previous credit
- AMT_PAYMENT : What the client actually paid on previous credit on this installment
- NAME_INCOME_TYPE : Clients income type
- OCCUPATION_TYPE : What kind of occupation does the client have
'''

INFO_VAR_PERF =  '''Détail des variables choisies pour la performance :
- AMT_CREDIT_SUM_DEBT : Current debt on Credit Bureau credit
- AMT_CREDIT_SUM : Current credit amount for the Credit Bureau credit
- EXT_SOURCE_1 : Normalized score from external data source 1
- EXT_SOURCE_2 : Normalized score from external data source 2
- EXT_SOURCE_3 : Normalized score from external data source 3
- NAME_INCOME_TYPE : Clients income type
- DAYS_EMPLOYED : How many days before the application the person started current employment
'''

def register_callbacks(app):
    @app.callback(
        [Output('result-output', 'children'),
         Output('app-tabs', 'value')],
        [Input('launch-button', 'n_clicks'),
         Input("variables-dropdown", 'value'),
         Input('model-choice', 'value')],
        prevent_initial_call=True
    )
    def update_result(n_clicks, features, model):
        if n_clicks and n_clicks > 0:
            if model is None or features is None :
                return dash.no_update

            results = html.Div([html.H3(f'Résultats {features, model}')])
            return (results, 'tab2')

        return dash.no_update

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
            new_values = ['REGION_RATING_CLIENT_W_CITY', 'DAYS_CREDIT_ENDDATE', 'RATE_DOWN_PAYMENT',
                          'AMT_PAYMENT', 'NAME_INCOME_TYPE', 'OCCUPATION_TYPE']
            info_text = INFO_VAR_INT

        elif button_id == 'performance-button':
            new_values = ['AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM', 'EXT_SOURCE_2',
                          'EXT_SOURCE_1', 'EXT_SOURCE_3', 'NAME_INCOME_TYPE',
                          "DAYS_EMPLOYED"]
            info_text = INFO_VAR_PERF

        else:
            new_values = existing_values

        return new_values, info_text
