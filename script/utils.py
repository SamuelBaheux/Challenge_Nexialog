import pandas as pd


class GridScore():
    def __init__(self, df, model):
        self.df = df
        self.model = model

    def calculate_percentage_default(self, row, data_frame):
        variable = row['Variable']
        modality = row['Modality']
        if modality.isdigit():
            modality = int(modality)

        default_count = data_frame[data_frame[variable] == modality]["TARGET"].sum()
        total_count = data_frame.shape[0]
        return round((default_count / total_count) * 100, 2)

    def calculate_pcentage_class(self, row, data_frame):
        variable = row['Variable']
        modality = row['Modality']
        if modality.isdigit():
            modality = int(modality)

        default_count = data_frame[data_frame[variable] == modality].shape[0]
        total_count = data_frame.shape[0]
        return round((default_count / total_count) * 100, 2)

    def compute_grid_score(self):
        results_summary_frame = self.model.summary2().tables[1]

        coefs = results_summary_frame['Coef.']
        p_values = results_summary_frame['P>|z|']

        max_coef = coefs.loc[coefs.index != 'Intercept'].max()
        min_coef = coefs.loc[coefs.index != 'Intercept'].min()

        score_card = pd.DataFrame(columns=['Variable', 'Coefficient', 'P-Value', 'Score'])

        for variable in coefs.index[1:]:
            coef = round(coefs[variable], 2)
            p_value = round(p_values[variable], 4)
            score = round(abs(max_coef - coef) / (max_coef - min_coef) * 1000, 2)

            score_card.loc[len(score_card)] = [variable, coef, p_value, score]

        score_card['Normalized Score'] = round((score_card['Score'] - score_card['Score'].min()) / (
                    score_card['Score'].max() - score_card['Score'].min()) * 1000, 2)

        score_card["Modality"] = score_card["Variable"].apply(lambda x: x.split("[T.")[1][:-1])
        score_card["Variable"] = score_card["Variable"].apply(lambda x: (x.split("C(")[1]).split(")")[0])

        score_card["Pcentage_Défaut"] = score_card.apply(lambda row: self.calculate_percentage_default(row, self.df), axis=1)
        score_card["Pcentage_Classe"] = score_card.apply(lambda row: self.calculate_pcentage_class(row, self.df), axis=1)

        score_card = score_card[
            ['Variable', "Modality", 'Coefficient', 'P-Value', 'Normalized Score', "Pcentage_Défaut",
             "Pcentage_Classe"]]

        return (score_card)
