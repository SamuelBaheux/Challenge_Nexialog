import jenkspy

from logit_utils import *
from xgb_utils import *

class Modelization():
    def init_model(self, model):
        if model == 'logit':
            self.model = LogitModel()
            self.model_name = "logit"
        else :
            self.model = XGB_model()
            self.model_name = "xgb"

    def init_data(self, data, intervalles_dic, target, date):
        self.model.init_data(data, intervalles_dic, target, date)

    def run_model(self):
        self.results = self.model.run_model()

    def get_metrics(self):
        return(self.model.get_metrics())

    def get_grid_score(self, train_prepared, target):
        if self.model_name == 'logit':
            GS = GridScore(train_prepared, self.results, target)
            print("Calcul de la grille de score ✅")
            self.grid_score = GS.compute_grid_score()
            self.df_score = GS.get_individual_score()
            self.grid_score["Variable"] = self.grid_score["Variable"].apply(lambda x: x.split("_disc_int")[0])
            self.grid_score["Variable"] = self.grid_score["Variable"].apply(lambda x: x.split("_discrete")[0])
            return(self.grid_score)

        else :
            print(self.results)
            gs = GridScoreXGB(train_prepared, self.results, target)
            print("Calcul de la grille de score ...")
            self.grid_score = gs.compute_grid_score()
            self.df_score = gs.get_individual_score()
            self.grid_score["Variable"] = self.grid_score["Variable"].apply(lambda x: x.split("_disc_int")[0])
            self.grid_score["Variable"] = self.grid_score["Variable"].apply(lambda x: x.split("_discrete")[0])
            return(self.grid_score)

    def get_segmentation(self, target):
        scores_clients = self.df_score["Score_ind"].sample(30000, replace = False)
        nombre_de_classes = 6

        print("Segmentation en cours ... ")
        breaks = jenkspy.jenks_breaks(scores_clients, nombre_de_classes)
        breaks = [round (score) for score in breaks]
        breaks[-1] = breaks[-2] + 50

        print(breaks)

        self.df_score["Classes"] = np.digitize(self.df_score["Score_ind"], bins=sorted(breaks))

        self.resultats = self.df_score.groupby("Classes").agg(
            Taux_Défaut=(target, "mean"),
            Population=(target, "size")
        )
        self.resultats['Taux_Individus'] = (self.resultats['Population'] / self.df_score.shape[0]) * 100

        return(self.resultats)

    def get_segmentation_metrics(self, target, date):
        stability_df = self.df_score.groupby([date, 'Classes'])[target].mean().unstack()
        count_seg = sum(stability_df[i + 1].max() > stability_df[i].min() for i in range(1, 7))

        self.resultats['defaut_monotone'] = (self.resultats['Taux_Défaut'].diff() < 0).astype(int)
        nombre_defauts_monotones = self.resultats['defaut_monotone'].sum() + 1

        count_seg = (count_seg / 7 * 100)
        nombre_defauts_monotones = (nombre_defauts_monotones / 7 * 100)

        return({"count_seg" : count_seg,
                "count_monotonie" : nombre_defauts_monotones})

    def get_moc_c(self, target):
        resultats = self.df_score.groupby("Classes").agg(moyenne_TARGET=(target, "mean")).to_dict()["moyenne_TARGET"]

        def Boostrapping_Classes(classe):
            df_classe = self.df_score[self.df_score['Classes'] == classe]
            tx_defaut_liste = []

            for _ in range(1000):
                echantillon = df_classe[target].sample(n=len(df_classe), replace=True)
                tx_defaut_liste.append(echantillon.mean())

            tx_defaut_liste = np.array(tx_defaut_liste)

            decile9 = np.percentile(tx_defaut_liste, 90)

            MOC_C = decile9 - resultats[classe]

            return (classe, resultats[classe], decile9, MOC_C)

        self.MOC_C = pd.DataFrame(columns=["Classe", "LRA", "Moc_C"])

        classes = list(self.df_score["Classes"].unique())
        classes.sort()

        for i in classes:
            classe, LRA, _, MOC_C_classe = Boostrapping_Classes(i)
            self.MOC_C.loc[len(self.MOC_C)] = [classe, LRA, MOC_C_classe]

        return(self.MOC_C)

