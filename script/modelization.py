import jenkspy

from Logit_utils import *
from XGB_utils import *

class Modelization():
    def init_model(self, model):
        if model == 'logit':
            self.model = LogitModel()
            self.model_name = "logit"
        else :
            self.model = XGB_model()
            self.model_name = "xgb"

    def init_data(self, data, intervalles_dic):
        self.model.init_data(data, intervalles_dic)

    def run_model(self):
        self.results = self.model.run_model()

    def get_metrics(self):
        return(self.model.get_metrics())

    def get_grid_score(self, train_prepared):
        if self.model_name == 'logit':
            GS = GridScore(train_prepared, self.results)
            print("Calcul de la grille de score ✅")
            grid_score = GS.compute_grid_score()
            self.df_score = GS.get_individual_score()
            return(grid_score)

        else :
            gs = GridScoreXGB(train_prepared, self.results)
            print("Calcul de la grille de score ...")
            grid_score = gs.compute_grid_score()
            self.df_score = gs.get_individual_score()
            return(grid_score)

    def get_segmentation(self):
        scores_clients = self.df_score["Score_ind"].sample(20000, replace = True)
        nombre_de_classes = 6

        print("Segmentation en cours ... ")
        breaks = jenkspy.jenks_breaks(scores_clients, nombre_de_classes)
        breaks = [round (score) for score in breaks]
        breaks[-1] = breaks[-2] + 50

        print(breaks)

        self.df_score["Classes"] = np.digitize(self.df_score["Score_ind"], bins=sorted(breaks))

        resultats = self.df_score.groupby("Classes").agg(
            moyenne_TARGET=("TARGET", "mean"),
            nb_individus=("TARGET", "size")
        )
        resultats['taux_individus'] = (resultats['nb_individus'] / self.df_score.shape[0]) * 100

        return(resultats)

