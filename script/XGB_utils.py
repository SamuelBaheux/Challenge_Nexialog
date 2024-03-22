import re

import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap
import warnings

import numpy as np

from script.data_preparation import ConstantFeatures

warnings.filterwarnings('ignore', category=FutureWarning,
                        message="Series.__getitem__ treating keys as positions is deprecated")

warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)




class GridScoreXGB():
    def __init__(self, df, shap_df):
        self.df = df
        self.shap_df = shap_df
        self.variable_pattern = r'C\(\s*([^,]+?)(?=,|\))'
        self.reference_pattern = r'Treatment\(reference="([^"]+)"\)'

    def calculate_percentage_default(self, row, data_frame):
        variable = row['Variable']

        if variable == "Intercept":
            return (0)

        modality = row['Modality']
        if modality.isdigit():
            modality = int(modality)

        if '_ref' in modality:
            modality = modality.split('_ref')[0]

        default_count = data_frame[data_frame[variable] == modality]["TARGET"].sum()
        total_count = data_frame.shape[0]
        return round((default_count / total_count) * 100, 2)

    def calculate_pcentage_class(self, row, data_frame):
        variable = row['Variable']
        if variable == "Intercept":
            return (0)

        modality = row['Modality']
        if modality.isdigit():
            modality = int(modality)

        if '_ref' in modality:
            modality = modality.split('_ref')[0]

        default_count = data_frame[data_frame[variable] == modality].shape[0]
        total_count = data_frame.shape[0]
        return round((default_count / total_count) * 100, 2)

    def calculate_contribution(self, score_card):
        mean_scores = score_card.groupby('Variable')['Score'].mean().to_dict()
        score_card['Contribution'] = score_card.apply(
            lambda x: (x["Pcentage_Classe"] / 100) * (x["Score"] - mean_scores.get(x["Variable"], 0)) ** 2,
            axis=1)
        contributions = np.sqrt(score_card.groupby('Variable')['Contribution'].sum()).to_dict()

        contrib_totale = sum(contributions.values())
        pcentage_contrib = {key: value / contrib_totale for key, value in contributions.items()}
        score_card['Contribution'] = score_card.apply(lambda x: round(pcentage_contrib[x["Variable"]], 2) * 100, axis=1)

        return (score_card)

    def compute_score(self, row):
        num = np.abs(self.max[row["Variable"]] - row["Coefficient"])
        denominateur = sum(self.max[key] - self.min[key] for key in self.min)
        return ((num / denominateur) * 1000)

    def extract_modality(self, reference_string):
        modality_pattern = r'\[T\.([^\]]+\]?)'
        match = re.search(modality_pattern, reference_string)

        if match:
            modality = match.group(1)
            if modality[-1] == "]" and not re.search(r'\d+\]$', modality):
                return modality[:-1]
            return modality
        else:
            return 'N/A'

    def compute_grid_score(self):
        score_card = self.shap_df.copy()

        self.max = score_card.groupby("Variable")["Coefficient"].max().to_dict()
        self.min = score_card.groupby("Variable")["Coefficient"].min().to_dict()

        score_card["Score"] = 0
        score_card["Score"] = score_card.apply(lambda x: self.compute_score(x), axis=1)

        score_card["Pcentage_Défaut"] = score_card.apply(lambda row: self.calculate_percentage_default(row, self.df),
                                                         axis=1)
        score_card["Pcentage_Classe"] = score_card.apply(lambda row: self.calculate_pcentage_class(row, self.df),
                                                         axis=1)

        score_card = self.calculate_contribution(score_card)

        self.score_card = score_card[['Variable', "Modality", 'Coefficient', "Score",
                                      "Contribution", "Pcentage_Défaut", "Pcentage_Classe"]]

        return self.score_card

    def compute_individual_score(self, row, features):
        score = 0
        for var in features:
            modality = row[var]
            score += self.score_dict[var][modality]
        return (score)

    def get_individual_score(self):
        self.score_dict = {}
        for index, row in self.score_card.iterrows():
            var = row["Variable"]
            mod = row["Modality"].split("_ref")[0]
            score = row["Score"]

            if var not in self.score_dict:
                self.score_dict[var] = {}
            self.score_dict[var][mod] = score

        features = list(self.score_dict.keys())

        df_score = self.df.copy()
        df_score["Score_ind"] = 0
        df_score["Score_ind"] = df_score.apply(lambda row: self.compute_individual_score(row, features), axis=1)
        return (df_score)

class XGB_model():
    def __init__(self):
        self.reference_dic = ConstantFeatures().dic_ref

    def init_data(self, train, intervalles_dic):
        self.train = train
        self.intervalles_dic = intervalles_dic

    def get_features_list(self):
        self.features = self.train.columns.to_list()
        self.features.remove("TARGET")

        try:
            self.features.remove("date_trimestrielle")
        except:
            pass

        try:
            self.features.remove("date_mensuelle")
        except:
            pass

    def get_ref_vars(self):
        self.dic_ref = {}

        for key, value in self.reference_dic.items():
            if key.split("_discret")[0] in self.features or key.split("_disc_int")[0] in self.features :
                if value == 'max':
                    self.dic_ref[key] = self.intervalles_dic[key.split("_disc_int")[0]][
                        list(self.intervalles_dic[key.split("_disc_int")[0]])[-1]]
                elif value == 'min':
                    self.dic_ref[key] = self.intervalles_dic[key.split("_disc_int")[0]][
                        list(self.intervalles_dic[key.split("_disc_int")[0]])[0]]
                else:
                    self.dic_ref[key] = value

    def get_dic_notref(self):
        self.dic_not_ref = {}
        for var in self.dic_ref.keys():
            modality = self.train[self.train[var] != self.dic_ref[var]][var].mode()[0]
            self.dic_not_ref[var] = modality

    def prepare_data(self):
        self.new_var = {}
        translation_table = str.maketrans({'[': 'zz',
                                           ']': 'vv',
                                           ';': 'ww',
                                           '-': 'ff',
                                           '.': 'pp'})

        for var in self.dic_ref.keys():
            self.new_var[f'{var}_{self.dic_ref[var]}'] = f'{var}_{self.dic_ref[var]}'.translate(translation_table)
            self.train.loc[:, f'{var}_{self.dic_ref[var]}'] = self.train[var].apply(
                lambda x: 0 if x == self.dic_ref[var] else 1)
            self.train = self.train.rename(
                columns={f'{var}_{self.dic_ref[var]}': self.new_var[f'{var}_{self.dic_ref[var]}']})

    def split_data(self):
        X = self.train[self.new_var.values()]
        y = self.train["TARGET"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42,
                                                                                stratify=y)

    def train_model(self):
        params = {'max_depth': 10, 'eta': 0.06739876573943267, 'gamma': 1.0887983081146109e-05,
                  'colsample_bytree': 0.5551288232191764, 'subsample': 0.9618418605416359, 'n_estimators': 97,
                  'alpha': 0.48989283547214435}
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(self.X_train, self.y_train)

    def get_metrics(self):
        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(self.y_test, y_prob)
        self.roc_auc = auc(fpr, tpr)
        self.gini = 2 * self.roc_auc - 1

        return({"roc_auc" : self.roc_auc,
                "gini" : self.gini,
                "fpr" : fpr,
                "tpr" : tpr})

    def compute_shap_values(self):
        explainer = shap.TreeExplainer(self.model)

        self.shap_values = explainer.shap_values(self.X_train)
        #shap.summary_plot(self.shap_values, self.X_train)

        # shap_values_single = explainer.shap_values(self.X_train.iloc[[0]])
        # shap.force_plot(explainer.expected_value, shap_values_single, self.X_train.iloc[[0]])

    def extract_features_name(self, string):
        bracket_content = re.search(r'(\[[^\]]+\])', string)
        if bracket_content:
            return (string.split(bracket_content.group(1))[0][:-1])
        else:
            last_part = re.search(r'[^_]+_[^_]+$', string)
            if last_part:
                return (string.split(last_part.group(0))[0][:-1])

    def get_modality(self, row):
        var = row["Var_ref"]
        if '_ref' in var:
            return (self.dic_ref[var.split('_ref')[0]] + '_ref')
        else:
            return (self.dic_not_ref[var])

    def get_shap_coef(self):
        shaps = pd.DataFrame(self.shap_values, columns=self.X_train.columns)
        shaps['TARGET'] = self.y_train.values

        shapley_values = pd.DataFrame(columns=['Coef', 'Variable'])

        dic_temp = {value: key for key, value in self.new_var.items()}
        for col in self.X_train.columns:
            shapley_values.loc[len(shapley_values)] = [shaps[shaps[col] < 0][col].mean(), dic_temp[col]]
            shapley_values.loc[len(shapley_values)] = [shaps[shaps[col] > 0][col].mean(), dic_temp[col]]

        shapley_values["Var_ok"] = shapley_values["Variable"].apply(lambda x: self.extract_features_name(x))

        unique_var_ok_indexes = shapley_values.drop_duplicates('Var_ok', keep='first').index
        shapley_values["Var_ref"] = shapley_values["Var_ok"].copy()
        shapley_values.loc[unique_var_ok_indexes, 'Var_ref'] += '_ref'
        shapley_values["Modality"] = shapley_values.apply(lambda x: self.get_modality(x), axis=1)
        shapley_values["Var_ref"] = shapley_values["Var_ref"].apply(lambda x: x.split("_ref")[0])

        shapley_values = shapley_values[["Coef", "Var_ref", "Modality"]]
        shapley_values = shapley_values.rename(columns={"Coef": "Coefficient",
                                                        "Var_ref": "Variable"})

        return (shapley_values)

    def run_model(self):
        self.get_features_list()
        self.get_ref_vars()
        self.get_dic_notref()
        self.prepare_data()
        self.split_data()
        self.train_model()
        self.compute_shap_values()
        return (self.get_shap_coef())