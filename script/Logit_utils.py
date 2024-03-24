import warnings
from data_preparation import ConstantFeatures

from sklearn.metrics import roc_curve, auc, f1_score, accuracy_score, precision_score
from statsmodels.discrete.discrete_model import Logit
import re
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=FutureWarning,
                        message="Series.__getitem__ treating keys as positions is deprecated")


class LogitModel():
    def __init__(self):
        self.reference_dic = ConstantFeatures().dic_ref

    def init_data(self, df, intervalles_dic):
        self.df = df
        self.intervalles_dic = intervalles_dic

    def get_features_list(self):
        self.features = self.df.columns.to_list()

        self.features.remove("date_mensuelle")
        self.features.remove("TARGET")
        try:
            self.features.remove("date_trimestrielle")
        except:
            pass

    def get_ref_vars(self):
        self.reference_features = {}

        for key, value in self.reference_dic.items():
            if key.split("_discret")[0] in self.features or key.split("_disc_int")[0] in self.features :
                if value == 'max':
                    self.reference_features[key] = self.intervalles_dic[key.split("_disc_int")[0]][
                        list(self.intervalles_dic[key.split("_disc_int")[0]])[-1]]
                elif value == 'min':
                    self.reference_features[key] = self.intervalles_dic[key.split("_disc_int")[0]][
                        list(self.intervalles_dic[key.split("_disc_int")[0]])[0]]
                else:
                    self.reference_features[key] = value

    def train_test_split(self):
        self.df_train = self.df.iloc[:280000, :]
        self.df_test = self.df.iloc[280000:, :]

    def get_formula(self):
        formula_parts = []

        for var in self.features:
            if var in self.reference_features:
                ref_category = self.reference_features[var]
                formula_parts.append(f'C({var}, Treatment(reference="{ref_category}"))')

        self.formula = 'TARGET ~ ' + ' + '.join(formula_parts)

    def train_logit(self):
        model = Logit.from_formula(formula=self.formula, data=self.df_train)
        self.logit_model = model.fit()

    def get_metrics(self):
        pred = self.logit_model.predict(self.df_test)
        fpr, tpr, thresholds = roc_curve(self.df_test["TARGET"], pred)

        roc_auc = auc(fpr, tpr)
        gini_coefficient = 2 * roc_auc - 1
        f1 = f1_score(self.df_test["TARGET"], round(pred)) * 100
        acc = accuracy_score(self.df_test["TARGET"], round(pred)) * 100
        prec = precision_score(self.df_test["TARGET"], round(pred)) * 100
        print(f1, acc, prec)
        return({"fpr": fpr,
                'tpr' : tpr,
                "f1": f1,
                "accuracy": acc,
                "precision": prec,
                'roc_auc' : roc_auc,
                'gini' : gini_coefficient})

    def run_model(self):
        self.get_features_list()
        self.get_ref_vars()
        self.train_test_split()
        self.get_formula()
        self.train_logit()
        return(self.logit_model)


class GridScore():
    def __init__(self, df, model):
        self.df = df
        self.model = model
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
        results_summary_frame = self.model.summary2().tables[1]

        coefs = results_summary_frame['Coef.']
        p_values = results_summary_frame['P>|z|']

        score_card = pd.DataFrame(columns=['Variable', 'Modality', 'Coefficient', 'P-Value'])
        score_card.loc[len(score_card)] = [coefs.index[0], '-', coefs[0], p_values[0]]

        previous_var_name = None
        for variable in coefs.index[1:]:
            coef = round(coefs[variable], 2)
            p_value = round(p_values[variable], 4)

            reference_match = re.search(self.reference_pattern, variable)
            reference = reference_match.group(1) if reference_match else "N/A"

            variable_name_match = re.search(self.variable_pattern, variable)
            variable_name = variable_name_match.group(1) if variable_name_match else variable

            if variable_name != previous_var_name:
                score_card.loc[len(score_card)] = [variable_name, reference + '_ref', 0, 0]

            modality = self.extract_modality(variable)

            score_card.loc[len(score_card)] = [variable_name, modality, coef, p_value]
            previous_var_name = variable_name

        self.max = score_card.groupby("Variable")["Coefficient"].max().to_dict()
        self.min = score_card.groupby("Variable")["Coefficient"].min().to_dict()

        score_card["Score"] = 0
        score_card["Score"] = score_card.apply(lambda x: self.compute_score(x), axis=1)

        score_card["Pcentage_Défaut"] = score_card.apply(lambda row: self.calculate_percentage_default(row, self.df),
                                                         axis=1)
        score_card["Pcentage_Classe"] = score_card.apply(lambda row: self.calculate_pcentage_class(row, self.df),
                                                         axis=1)

        score_card = self.calculate_contribution(score_card)

        self.score_card = score_card[['Variable', "Modality", 'Coefficient', 'P-Value', "Score",
                                 "Contribution", "Pcentage_Défaut", "Pcentage_Classe"]]

        return self.score_card

    def compute_individual_score(self, row, features):
        score = 0
        for var in features :
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
        features.remove("Intercept")

        df_score = self.df.copy()
        df_score["Score_ind"] = 0
        df_score["Score_ind"] = df_score.apply(lambda row: self.compute_individual_score(row, features), axis=1)
        return (df_score)
