import pickle
import sys

sys.path.append('./script')

from dash.dependencies import Input, Output, State, ALL
import dash

from builders import *
from data_preparation import *
from plot_utils import *
from plot_analyse import *
from app_utils import *

def register_callbacks(app):

    ####################################### ANALYSE ########################################

    @app.callback([Output('output-data-upload-analyse', 'children'),
                   Output('target-dropdown-analyse', 'options'),
                   Output('date-dropdown-analyse', 'options')],
                  [Input('upload-data-analyse', 'contents')],
                  [State('upload-data-analyse', 'filename')])
    def update_output(contents, filename):
        if contents is not None:
            df, filename = parse_contents(contents, filename)
            if isinstance(df, pd.DataFrame):
                analyse.init_data(df)
                target_options = analyse.get_features()
                date_options = analyse.get_features()

                return [html.Div(html.H5(f'Le fichier {filename} a été téléchargé avec succès'))], target_options, date_options
        else:
            return dash.no_update,[], []

    @app.callback([Output("analyze_glob_data", "style", allow_duplicate=True),
                   Output("analyze_glob_data", "children", allow_duplicate=True),
                   Output("analyze_var_data", "children", allow_duplicate=True)],
                  [Input("target-dropdown-analyse", "value"),
                   Input("date-dropdown-analyse", "value"),
                   Input("launch-button-analyse", "n_clicks")],
                  prevent_initial_call=True)
    def display_graph(target, date, n_clicks):
        if n_clicks and n_clicks > 0:
            analyse.init_target(target)
            analyse.init_date(date)
            return {'display': 'block'}, build_analyse_data(), build_analyse_feature()
        else :
            return dash.no_update

    @app.callback([Output("analyze_glob_data", "style", allow_duplicate=True),
                   Output("analyze_var_data", "style", allow_duplicate=True)],
                  [Input("analyse-var-button", "n_clicks")],
                  prevent_initial_call=True)
    def render(n_clicks):
        if n_clicks and n_clicks > 0:
            return {'display': 'None'},{'display': 'block'}
        else :
            return dash.no_update

    @app.callback([Output("analyze_glob_data", "style", allow_duplicate=True),
                   Output("analyze_var_data", "style", allow_duplicate=True)],
                  [Input("analyse-global-button", "n_clicks")],
                  prevent_initial_call=True
                  )
    def render(n_clicks):
        if n_clicks and n_clicks > 0:
            return {'display': 'block'},{'display': 'None'}
        else :
            return dash.no_update

    @app.callback([Output("plot_stability", "figure"),
                   Output("plot_distrib", "figure"),
                   Output("variables-info-analyse", "children")],
                  [Input("target-dropdown-analyse-var", "value")])
    def render(var):
        return plot_stability_analyse(var), plot_marginal_density(var), [dcc.Markdown(texte_analyse_var(var))]


    ####################################### MODÉLISATION ########################################
    @app.callback([Output('output-data-upload', 'children'),
                   Output('variables-dropdown', 'options'),
                   Output('target-dropdown', 'options'),
                   Output('date-dropdown', 'options'),
                   Output('predefined_vars_button', 'style')],
                  [Input('upload-data', 'contents')],
                  [State('upload-data', 'filename')])
    def update_output(list_of_contents, list_of_names):
        if list_of_contents is None or not list_of_contents:
            return [html.Div(''), [], [], [], {'display': 'None'}]

        if not isinstance(list_of_contents, list):
            list_of_contents = [list_of_contents]
        if not isinstance(list_of_names, list):
            list_of_names = [list_of_names]

        children = []
        options = []
        target_options = []
        date_options = []

        for c, n in zip(list_of_contents, list_of_names):
            results = parse_contents(c, n)
            if results is not None:
                dataprep.initialize_df(results[0])
                options = dataprep.get_features()
                target_options = dataprep.get_features()
                date_options = dataprep.get_features()
                children = [html.Div(html.H5(f'Le fichier {results[1]} a été téléchargé avec succès'))]

        if list_of_names[0] == "application_train_vf.csv" :
            return children, options, target_options, date_options, {'display': 'flex'}
        else :
            return children, options, target_options, date_options, {'display': 'None'}

    @app.callback(Output('hidden-div', 'children'),
                  [Input('target-dropdown', 'value')])
    def update_target(target_selected):
        if target_selected is not None :
            dataprep.init_target(target_selected)
            return ''

    @app.callback(Output('hidden-div1', 'children'),
                  [Input('date-dropdown', 'value')])
    def update_target(date_selected):
        if date_selected is not None :
            dataprep.init_date(date_selected)
            return ''


    @app.callback(
        [Output('app-tabs', 'value'),
         Output('Control-chart-tab', 'style'),
         Output("loading-output", "children"),
         Output('Control-chart-tab', 'children'),
         Output("chat-tab", "style"),
         Output("chat-tab", "children"),
         Output("Control-chart-tab-2", "style"),
         Output("Control-chart-tab-2", "children"),
         Output("denotching-tab", "style"),
         Output("denotching-tab", "children"),
         ],
        [Input('launch-button', 'n_clicks'),
         Input("variables-dropdown", 'value'),
         Input('model-choice', 'value')],
        prevent_initial_call=True)
    def update_result(n_clicks, features, model_choice):
        if n_clicks and n_clicks > 0:
            if model_choice is None or features is None:
                return dash.no_update

            print("Initialisation des données")
            dataprep.initialize_data(features)
            train_prepared = dataprep.get_prepared_data()

            print("Entraînement du modèle")
            if model_choice == 'Logit':
                model_classique.init_model('logit')
                model_classique.init_data(train_prepared, dataprep.discretizer.intervalles_dic, dataprep.target, dataprep.date)
                model_classique.run_model()
                model_classique.get_grid_score(dataprep.train, dataprep.target)
                model_classique.get_segmentation(dataprep.target)
                model_classique.get_default_proba(dataprep.target, dataprep.date)
                dataprep.init_model_name('logit')

                return ('tab2', {"display": "flex"}, "loaded", build_logit_model(), {"display": "flex"}, chatbot(), {"display": "none"}, dash.no_update, {"display": "flex"}, layout_denot())

            elif model_choice == 'XGBoost':
                model_challenger.init_model('xgb')
                model_challenger.init_data(train_prepared, dataprep.discretizer.intervalles_dic, dataprep.target, dataprep.date)
                model_challenger.run_model()
                model_challenger.get_grid_score(dataprep.train, dataprep.target)
                model_challenger.get_segmentation(dataprep.target)
                model_challenger.get_default_proba(dataprep.target, dataprep.date)
                dataprep.init_model_name('xgb')
                return ('tab3', {"display": "none"}, "loaded", dash.no_update, {"display": "flex"}, chatbot(), {"display": "flex"},  build_xgboost_model(), {"display": "flex"}, layout_denot())

            elif model_choice == "both":
                model_classique.init_model('logit')
                model_classique.init_data(train_prepared, dataprep.discretizer.intervalles_dic, dataprep.target, dataprep.date)
                model_classique.run_model()
                model_classique.get_grid_score(dataprep.train, dataprep.target)
                model_classique.get_segmentation(dataprep.target)
                model_classique.get_default_proba(dataprep.target, dataprep.date)

                model_challenger.init_model('xgb')
                model_challenger.init_data(train_prepared, dataprep.discretizer.intervalles_dic, dataprep.target, dataprep.date)
                model_challenger.run_model()
                model_challenger.get_grid_score(dataprep.train, dataprep.target)
                model_challenger.get_segmentation(dataprep.target)
                model_challenger.get_default_proba(dataprep.target, dataprep.date)

                dataprep.init_model_name('logit')

                logit_layout, xgb_layout = build_both_model()
                return ('tab2', {"display": "flex"}, "loaded", logit_layout, {"display": "flex"}, chatbot(), {"display": "flex"}, xgb_layout, {"display": "flex"}, layout_denot())



        return dash.no_update, {"display": "none"}, dash.no_update, dash.no_update,  {"display": "none"}, dash.no_update, {"display": "none"}, dash.no_update, {"display": "none"}, dash.no_update

    @app.callback(
        Output('loading-div', 'style'),
        [Input('launch-button', 'n_clicks'),
         Input("variables-dropdown", 'value'),
         Input('model-choice', 'value')],
        [State('loading-div', 'style')],
        prevent_initial_call=True
    )
    def toggle_loading_div(n_clicks, features, model_choice, current_style):
        if n_clicks and n_clicks > 0:
            if model_choice is None or features is None:
                return dash.no_update
            return {'display': 'block'}
        return {'display': 'none'}

    @app.callback(
        Output('variables-dropdown', 'value'),
        [Input('interpretabilite-button', 'n_clicks'),
         Input('performance-button', 'n_clicks')],
        [State('variables-dropdown', 'value')]
    )
    def update_dropdown_variables(interpretabilite_n_clicks, performance_n_clicks, existing_values):
        ctx = dash.callback_context

        if not ctx.triggered:
            button_id = 'No clicks yet'
        else:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if button_id == 'interpretabilite-button':
            new_values = ['REGION_RATING_CLIENT_W_CITY', 'DAYS_CREDIT_ENDDATE', 'RATE_DOWN_PAYMENT',
                          'AMT_PAYMENT', 'NAME_INCOME_TYPE', 'OCCUPATION_TYPE']

        elif button_id == 'performance-button':
            new_values = ['AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM', 'EXT_SOURCE_2',
                          'EXT_SOURCE_1', 'EXT_SOURCE_3', 'NAME_INCOME_TYPE',
                          "DAYS_EMPLOYED"]

        else:
            new_values = existing_values

        return new_values

    @app.callback(
        Output('variables-info-markdown', 'children'),
        [Input('variables-dropdown', 'value')]
    )
    def update_dropdown_variables(selected_features):
        if selected_features is not None :
            info = "**Détail des variables choisies pour la modélisation :**\n"
            for features in selected_features :
                desc = dictionnaire[dictionnaire['Row'] == features]['Description']
                if len(desc) > 0 :
                    info += f"- **{features}** : {desc.values[0]}\n"
                else :
                    info += f"- {features}\n"

            return info


    @app.callback(
        [Output('class-display-clas', 'figure', allow_duplicate=True),  # Mise à jour de la figure du graphique
         Output('table-id-clas', 'data')],
        [Input('breaks-slider-clas', 'value'),
         Input('graph-type-selector-clas', 'value')],
        prevent_initial_call = True
    )
    def update_breaks_graph(breaks, graph_type):
        if breaks is not None :
            model_classique.update_segmentation(breaks, dataprep.target)

            if graph_type == 'gini':
                fig = create_gini_figure(model_classique)
            elif graph_type == 'taux':
                fig = create_stability_figure(model_classique)

            table_data = round(model_classique.resultats, 2).to_dict('records')

            return fig, table_data

    @app.callback(
        [Output('class-display-chal', 'figure', allow_duplicate=True),  # Mise à jour de la figure du graphique
         Output('table-id-chal', 'data')],
        [Input('breaks-slider-chal', 'value'),
         Input('graph-type-selector-chal', 'value')],
        prevent_initial_call = True
    )
    def update_breaks_graph(breaks, graph_type):
        if breaks is not None :
            model_challenger.update_segmentation(breaks, dataprep.target)

            if graph_type == 'gini':
                fig = create_gini_figure(model_challenger)
            elif graph_type == 'taux':
                fig = create_stability_figure(model_challenger)

            table_data = round(model_challenger.resultats, 2).to_dict('records')

            return fig, table_data



    @app.callback(
        [Output('stability-graph-clas', 'figure'),
         Output('histo-graph-clas', 'figure')],
        [Input('stability-dropdown-clas', 'value')]
    )
    def update_graph(selected_variable):
        fig_stab = plot_stability_plotly(selected_variable)
        fig_hist = plot_hist(selected_variable)
        return [fig_stab, fig_hist]

    @app.callback(
        [Output('stability-graph-chal', 'figure'),
         Output('histo-graph-chal', 'figure')],
        [Input('stability-dropdown-chal', 'value')]
    )
    def update_graph(selected_variable):
        fig_stab = plot_stability_plotly(selected_variable)
        fig_hist = plot_hist(selected_variable)
        return [fig_stab, fig_hist]

    @app.callback(
        Output('class-display-clas', 'figure'),
        [Input('graph-type-selector-clas', 'value')]
    )
    def update_graph(selected_graph):
        if selected_graph == 'gini':
            fig = create_gini_figure(model_classique)
        elif selected_graph == 'taux':
            fig = create_stability_figure(model_classique)
        return fig

    @app.callback(
        Output('class-display-chal', 'figure'),
        [Input('graph-type-selector-chal', 'value')]
    )
    def update_graph(selected_graph):
        if selected_graph == 'gini':
            fig = create_gini_figure(model_challenger)
        elif selected_graph == 'taux':
            fig = create_stability_figure(model_challenger)
        return fig

    @app.callback(
        Output("download-df-score", "data"),
        Input("btn_df_score", "n_clicks"),
        prevent_initial_call=True,
    )
    def download_df_score(n_clicks):
        if dataprep.model_name == "logit" :
            model = model_classique
        elif dataprep.model_name == "xgb":
            model = model_challenger

        return dcc.send_data_frame(model.df_score.to_csv, "data_score.csv", index=False)

    @app.callback(
        Output("download-grille-score", "data"),
        Input("btn_grille_score", "n_clicks"),
        prevent_initial_call=True,
    )
    def download_grille_score(n_clicks):
        if dataprep.model_name == "logit" :
            model = model_classique
        elif dataprep.model_name == "xgb":
            model = model_challenger

        return dcc.send_data_frame(model.grid_score.to_csv, "grille_score.csv", index=False)

    @app.callback(
        Output("download-model", "data"),
        Input("btn_model", "n_clicks"),
        prevent_initial_call=True
    )
    def download_model(n_clicks):
        if dataprep.model_name == "logit" :
            model = model_classique
        elif dataprep.model_name == "xgb":
            model = model_challenger

        serialized_model = pickle.dumps(model.model)
        return dcc.send_bytes(serialized_model, "model.pickle")

    ####################################### CHATBOT ########################################
    @app.callback(
        Output('dynamic-radioitems-container', 'children'),
        [Input({'type': 'dynamic-radioitems', 'index': ALL}, 'value')],
        [State('dynamic-radioitems-container', 'children')]
    )
    def add_radioitems(values, children):
        if dataprep.model_name == "logit":
            model = model_classique
        elif dataprep.model_name == "xgb":
            model = model_challenger

        if not values or None in values:
            return dash.no_update

        df = model.df_score
        dropdown_columns = df.columns.difference(['Score_ind', 'Classes', dataprep.target, dataprep.date,
                                                  "date_trimestrielle"]).tolist()

        next_index = len(values)
        if next_index < len(dropdown_columns):
            new_element = html.Div([
                html.Img(src="./assets/images/robot.png", className="robot-img"),
                html.Div([
                    html.Label(format_option_column(dropdown_columns[next_index]),
                               className='label-inline message-label'),
                ], className='message-container'),
                html.Div([
                    dcc.RadioItems(
                        id={'type': 'dynamic-radioitems', 'index': next_index},
                        options=[
                            {'label': format_option_label(v), 'value': v}
                            for v in df[dropdown_columns[next_index]].dropna().unique()[::-1]
                        ],
                        labelStyle={'display': 'inline-block', 'margin-right': '20px'},
                        className='radio-inline selection-radio'
                    ),
                ], className='radioitems-container'),
            ], className='form-input row', style={'margin-bottom': '50px'})
            children.append(new_element)
        return children

    @app.callback(
        Output('score-ind-result', 'children'),
        [Input({'type': 'dynamic-radioitems', 'index': ALL}, 'value')],
        prevent_initial_call=True
    )
    def update_score_ind(dropdown_values):
        if None in dropdown_values:
            return dash.no_update

        df = model_classique.df_score if dataprep.model_name == "logit" else model_challenger.df_score
        dropdown_columns = df.columns.difference(
            ['Score_ind', 'Classes', dataprep.target, dataprep.date, "date_trimestrielle"]).tolist()

        if len(dropdown_values) < len(dropdown_columns):
            return dash.no_update

        model = model_classique if dataprep.model_name == "logit" else model_challenger
        filtered_df = df.copy()
        for column, value in zip(dropdown_columns, dropdown_values):
            if value is not None:
                filtered_df = filtered_df[filtered_df[column] == value]

        mean_score_ind = filtered_df['Score_ind'].mean() if not filtered_df.empty else None
        mean_classes = int(filtered_df['Classes'].mean()) if not filtered_df.empty else None

        message_lines = [
            f"Au vu de vos choix, votre score est {mean_score_ind:.2f}, vous êtes donc dans la classe {mean_classes}."
        ]

        if mean_classes > 5:
            message_lines.append("Un crédit vous sera octroyé.")
        elif 3 >= mean_classes >= 5:
            message_lines.append("Un crédit vous sera octroyé, mais avec un taux d'intérêt élevé.")
        else:
            message_lines.append("Malheureusement, aucun crédit ne vous sera octroyé.")

        message_divs = [html.Div(line, className='message-line') for line in message_lines]

        return html.Div([
            html.Img(src="./assets/images/robot.png", className="robot-img"),
            html.Div(message_divs, className='score-result-container'),
        ], style ={"color":"#FFFFFF", "display" :"flex"})

    ####################################### DENOTCHING ########################################

    @app.callback(
        [Output('button-25', 'className'),
         Output('button-50', 'className'),
         Output('button-75', 'className'),
         Output('button-100', 'className')],
        [Input('button-25', 'n_clicks'),
         Input('button-50', 'n_clicks'),
         Input('button-75', 'n_clicks'),
         Input('button-100', 'n_clicks')]
    )
    def update_button_styles(n_clicks_25, n_clicks_50, n_clicks_75, n_clicks_100):
        # Determine which button was clicked last
        ctx = dash.callback_context

        if not ctx.triggered:
            # No buttons have been clicked yet
            button_id = None
        else:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        # Update classes based on which button was clicked
        base_class = 'denot-button'
        active_class = 'denot-button-clicked'  # Add this class in your CSS

        return [
            active_class if button_id == 'button-25' else base_class,
            active_class if button_id == 'button-50' else base_class,
            active_class if button_id == 'button-75' else base_class,
            active_class if button_id == 'button-100' else base_class,
        ]

    @app.callback(
        [Output("compare_PD", "figure"),
         Output("compare_monotonie", "figure"),
         Output("compare_pop", "figure"),
         Output("denotching-graph", "style")],
        [Input('button-25', 'n_clicks'),
         Input('button-50', 'n_clicks'),
         Input('button-75', 'n_clicks'),
         Input('button-100', 'n_clicks')]
    )
    def render(n_clicks_25, n_clicks_50, n_clicks_75, n_clicks_100):
        ctx = dash.callback_context

        if not ctx.triggered:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
        else:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        ampleur = int(button_id.split("-")[1])

        model_classique.denotching(ampleur, dataprep.target, dataprep.date)

        return compare_PD(), compare_monotonie(), compare_pop(), {"display":"block"}
