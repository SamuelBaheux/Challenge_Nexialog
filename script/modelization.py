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

        else :
            gs = GridScoreXGB(train_prepared, self.results, target)
            print("Calcul de la grille de score ...")
            self.grid_score = gs.compute_grid_score()
            self.df_score = gs.get_individual_score()
            self.grid_score["Variable"] = self.grid_score["Variable"].apply(lambda x: x.split("_disc_int")[0])
            self.grid_score["Variable"] = self.grid_score["Variable"].apply(lambda x: x.split("_discrete")[0])
            self.grid_score["Coefficient"] = self.grid_score["Coefficient"].astype("float")

    def get_segmentation(self, target):
        scores_clients = self.df_score["Score_ind"].sample(30000, replace = False)
        print(self.df_score.columns)
        nombre_de_classes = 6

        print("Segmentation en cours ... ")
        self.breaks = jenkspy.jenks_breaks(scores_clients, nombre_de_classes)
        self.breaks = [round (score) for score in self.breaks]
        self.breaks[-1] = self.breaks[-2] + 50

        print(self.breaks)

        colonnes_int = ['date_mensuelle', 'TARGET', 'DAYS_CREDIT_ENDDATE_disc_int',
                              'RATE_DOWN_PAYMENT_disc_int', 'AMT_PAYMENT_disc_int',
                              'NAME_INCOME_TYPE_discret', 'OCCUPATION_TYPE_discret',
                              'REGION_RATING_CLIENT_W_CITY', 'Score_ind']

        colonnes_int_chal = ['date_mensuelle', 'TARGET', 'DAYS_CREDIT_ENDDATE_disc_int',
       'RATE_DOWN_PAYMENT_disc_int', 'AMT_PAYMENT_disc_int',
       'NAME_INCOME_TYPE_discret', 'OCCUPATION_TYPE_discret',
       'REGION_RATING_CLIENT_W_CITY', 'RATE_DOWN_PAYMENT_disc_int_[0.2;4.48]',
       'Score_ind']

        colonnes_perf = ['date_mensuelle', 'TARGET', 'AMT_CREDIT_SUM_disc_int',
                       'AMT_CREDIT_SUM_DEBT_disc_int', 'EXT_SOURCE_3_disc_int',
                       'DAYS_EMPLOYED_disc_int', 'EXT_SOURCE_2_disc_int',
                       'EXT_SOURCE_1_disc_int', 'NAME_INCOME_TYPE_discret', 'Score_ind']

        if set(self.df_score.columns) == set(colonnes_int) or set(self.df_score.columns) == set(colonnes_int_chal):
            self.breaks = [0.0, 110, 240, 340, 420, 550, 690]

        elif set(self.df_score.columns) == set(colonnes_perf) :
            self.breaks = [0.0, 210, 360, 460, 580, 670, 850.0]

        self.df_score["Classes"] = np.digitize(self.df_score["Score_ind"], bins=sorted(self.breaks))

        self.resultats = self.df_score.groupby("Classes").agg(
            Taux_Défaut=(target, "mean"),
            Population=(target, "size")
        )
        self.resultats['Taux_Individus'] = (self.resultats['Population'] / self.df_score.shape[0]) * 100
        self.resultats["CHR"] = range(1,self.resultats.shape[0]+1)
        self.resultats = self.resultats[['CHR', "Taux_Défaut", "Taux_Individus"]]

    def update_segmentation(self, new_breaks, target):
        self.df_score["Classes"] = np.digitize(self.df_score["Score_ind"], bins=sorted(new_breaks))

        self.resultats = self.df_score.groupby("Classes").agg(
            Taux_Défaut=(target, "mean"),
            Population=(target, "size")
        )
        self.resultats['Taux_Individus'] = (self.resultats['Population'] / self.df_score.shape[0]) * 100
        self.resultats["CHR"] = range(1,8)
        self.resultats = self.resultats[['CHR', "Taux_Défaut", "Taux_Individus"]]


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

    def get_moc_a(self, target, date):
        if 'date_trimestrielle' not in self.df_score.columns :
            self.df_score[date] = pd.to_datetime(self.df_score[date])
            self.df_score['date_trimestrielle'] = (self.df_score[date].dt.year.astype(str) + '_' +
                                                self.df_score[date].dt.quarter.astype(str))

        avant_covid = self.df_score[self.df_score['date_trimestrielle'] < '2020_2']

        taux_defaut_avant_covid = avant_covid.groupby('Classes')[target].mean()
        taux_defaut_pendant_covid = self.df_score.groupby('Classes')[target].mean()

        impact_covid_par_classe = (taux_defaut_avant_covid / taux_defaut_pendant_covid) - 1
        variance_taux_defaut_pendant_covid = self.df_score.groupby('Classes')[target].var()

        simulations = pd.DataFrame()

        for classe in impact_covid_par_classe.index:
            moyenne_impact = impact_covid_par_classe[classe]
            variance_impact = variance_taux_defaut_pendant_covid[classe]
            variance_impact = variance_impact if variance_impact > 0 else 0.001
            tirages = np.random.normal(moyenne_impact, variance_impact, 1000)
            simulations[classe] = tirages

        percentile_90 = simulations.quantile(0.90)
        self.MOC_A = percentile_90 - taux_defaut_pendant_covid
        self.MOC_A = self.MOC_A.reset_index()
        self.MOC_A.columns = ["CHR", "MOC_A"]
        self.MOC_A["MOC_A"] = self.MOC_A["MOC_A"].apply(lambda x : 0 if x < 0 else x)

        return(self.MOC_A)

    def get_default_proba(self, target, date):
        moc_c = self.get_moc_c(target)
        moc_a = self.get_moc_a(target, date)

        default_proba = moc_c.copy()
        default_proba["MOC_A"] = moc_a["MOC_A"]
        default_proba["Moc_C"] = default_proba["Moc_C"].apply(lambda x: 0 if x < 0 else x)
        default_proba["Probabilité_Défaut"] = default_proba["LRA"] + default_proba["Moc_C"] + default_proba["MOC_A"]
        self.default_proba = default_proba

    def denotching(self, ampleur, target, date):
        self.default_proba_before = self.default_proba.copy()
        self.default_proba_before["Taux_Individus"] = self.resultats["Taux_Individus"]

        new_breaks = [self.breaks[i] + [0,ampleur,ampleur,ampleur,ampleur,ampleur,ampleur][i] for i in range(len(self.breaks))]

        self.update_segmentation(new_breaks, target)
        self.get_default_proba(target, date)
        self.default_proba["Taux_Individus"] = self.resultats["Taux_Individus"]



