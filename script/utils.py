import pandas as pd
import numpy as np
import re
import warnings

warnings.filterwarnings('ignore', category=FutureWarning, message="Series.__getitem__ treating keys as positions is deprecated")


class GridScore():
    def __init__(self, df, model):
        self.df = df
        self.model = model
        self.variable_pattern = r'C\(([^,]+)'
        self.modality_pattern = r'\[T\.([^\]]+\])'
        self.reference_pattern = r'reference="([^"]+)"'

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
        mean_scores = score_card.groupby('Variable')['Normalized Score'].mean().to_dict()
        score_card['Contribution'] = score_card.apply(
            lambda x: (x["Pcentage_Classe"] / 100) * (x["Normalized Score"] - mean_scores.get(x["Variable"], 0)) ** 2,
            axis=1)
        contributions = np.sqrt(score_card.groupby('Variable')['Contribution'].sum()).to_dict()

        contrib_totale = sum(contributions.values())
        pcentage_contrib = {key: value / contrib_totale for key, value in contributions.items()}
        score_card['Contribution'] = score_card.apply(lambda x: round(pcentage_contrib[x["Variable"]], 2) * 100, axis=1)

        return (score_card)

    def compute_grid_score(self):
        results_summary_frame = self.model.summary2().tables[1]

        coefs = results_summary_frame['Coef.']
        p_values = results_summary_frame['P>|z|']

        max_coef = coefs.loc[coefs.index != 'Intercept'].max()
        min_coef = coefs.loc[coefs.index != 'Intercept'].min()

        score_card = pd.DataFrame(columns=['Variable', 'Modality', 'Coefficient', 'P-Value', 'Score'])
        score_card.loc[len(score_card)] = [coefs.index[0], '-', coefs[0], p_values[0], 0]

        previous_reference = None
        for variable in coefs.index[1:]:
            coef = round(coefs[variable], 2)
            p_value = round(p_values[variable], 4)
            score = round(abs(max_coef - coef) / (max_coef - min_coef) * 1000, 2)

            reference_match = re.search(self.reference_pattern, variable)
            reference = reference_match.group(1) if reference_match else "N/A"

            variable_name_match = re.search(self.variable_pattern, variable)
            variable_name = variable_name_match.group(1) if variable_name_match else variable

            if reference != previous_reference:
                score_card.loc[len(score_card)] = [variable_name, reference + '_ref', 0, 0, 0]

            modality_match = re.search(self.modality_pattern, variable)
            modality = modality_match.group(1) if modality_match else "N/A"

            score_card.loc[len(score_card)] = [variable_name, modality, coef, p_value, score]
            previous_reference = reference

        score_card['Normalized Score'] = round((score_card['Score'] - score_card['Score'].min()) / (
                score_card['Score'].max() - score_card['Score'].min()) * 1000, 2)

        score_card["Pcentage_Défaut"] = score_card.apply(lambda row: self.calculate_percentage_default(row, self.df),
                                                         axis=1)
        score_card["Pcentage_Classe"] = score_card.apply(lambda row: self.calculate_pcentage_class(row, self.df),
                                                         axis=1)

        score_card = self.calculate_contribution(score_card)

        score_card = score_card[
            ['Variable', "Modality", 'Coefficient', 'Contribution', 'P-Value', 'Normalized Score', "Pcentage_Défaut",
             "Pcentage_Classe"]]

        return score_card
