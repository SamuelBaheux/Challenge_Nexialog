import warnings
from joblib import Parallel, delayed
from functools import partial
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from scipy.stats import chi2_contingency
from tqdm import tqdm


class Genetic_Numerical_Discretisation():
    def __init__(self, train, variables_dict, plot=False):
        self.train = train
        self.variables_dict = variables_dict
        self.plot = plot

    def evalChi2(self, individual, variable, dataset):
        individual_sorted = sorted(individual)
        percentiles = [max(0, min(i * 100, 100)) for i in individual_sorted]
        thresholds = np.percentile(dataset[variable], percentiles)

        thresholds = np.unique(thresholds)

        disc_var = np.digitize(dataset[variable].dropna(), thresholds)
        dataset['disc_var'] = disc_var

        contingency_table = pd.crosstab(dataset['disc_var'], dataset['TARGET'])
        chi2, p, dof, expected = chi2_contingency(contingency_table)

        return chi2,

    def plot_stability(self, variable):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))

        stability_volume_df = self.train.groupby(['date_trimestrielle', variable]).size().unstack()

        for class_label in stability_volume_df.columns:
            values = stability_volume_df[class_label]
            axes[0].plot(stability_volume_df.index, values, label=f'Classe {class_label}', marker='o')

        axes[0].set_title(f'Stabilité de volume pour {variable}')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Nombre d\'observations')
        axes[0].legend(title='Classes de_binned', loc='upper left', bbox_to_anchor=(1, 1))
        axes[0].tick_params(axis='x', rotation=45)

        stability_taux_df = self.train.groupby(['date_trimestrielle', variable])['TARGET'].mean().unstack()
        stability_taux_df['stability'] = stability_taux_df.std(axis=1) / stability_taux_df.mean(axis=1)

        for class_label in stability_taux_df.drop('stability', axis=1).columns:
            values = stability_taux_df[class_label]
            axes[1].plot(stability_taux_df.index, values, label=f'Classe {class_label}', marker='o')

        axes[1].set_title(f'Stabilité de taux pour {variable}')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Proportion de la cible TARGET')
        axes[1].legend(title='Classes de_binned', loc='upper left', bbox_to_anchor=(1, 1))
        axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

    def genetic_discretisation(self, train_set, variable, nb_classes_max):
        temp = train_set[[variable, 'TARGET']].copy()
        NB_GEN = 5
        POP_SIZE = 100
        CXPB, MUTPB = 0.5, 0.2

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        def custom_crossover(ind1, ind2):
            if len(ind1) > 1 and len(ind2) > 1:
                return tools.cxTwoPoint(ind1, ind2)
            else:
                return ind1, ind2

        toolbox = base.Toolbox()
        toolbox.register("attr_float", np.random.uniform, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=nb_classes_max)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", partial(self.evalChi2, variable=variable, dataset=temp))
        toolbox.register("mate", custom_crossover)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)

        population = toolbox.population(n=POP_SIZE)
        algorithms.eaSimple(population, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=NB_GEN, verbose=False)

        best_ind = tools.selBest(population, 1)[0]
        return (best_ind)

    def discretize_variable(self, variable):
        print(variable)
        bins = self.genetic_discretisation(self.train, variable, self.variables_dict[variable] - 1)
        bins_normalise = sorted([min(1, max(0, val)) for val in bins])

        seuils = np.percentile(self.train[variable], [val * 100 for val in bins_normalise])
        seuils_uniques = np.unique(seuils)

        seuils_uniques = np.unique(seuils)
        intervalles = self.train.groupby(np.digitize(self.train[variable], seuils_uniques))[variable].agg(
            ['min', 'max'])
        dict_renommage = {modalite: f'[{round(row["min"], 2)};{round(row["max"], 2)}]' for modalite, row in
                          intervalles.iterrows()}

        return variable, seuils_uniques, dict_renommage

    def run_discretisation(self):
        self.train["date_mensuelle"] = pd.to_datetime(self.train["date_mensuelle"])
        self.train['date_trimestrielle'] = (self.train['date_mensuelle'].dt.year.astype(str) + '_' +
                                            self.train['date_mensuelle'].dt.quarter.astype(str))

        self.intervalles_dic = {}

        results = Parallel(n_jobs=-1)(delayed(self.discretize_variable)(variable) for variable in self.variables_dict)

        for variable, seuils_uniques, dict_renommage in results:
            self.train[f'{variable}_disc'] = np.digitize(self.train[variable], seuils_uniques)
            self.train[f'{variable}_disc_int'] = self.train[f'{variable}_disc'].map(dict_renommage)
            self.intervalles_dic[variable] = dict_renommage

        return self.train


if __name__ == '__main__' :
    train = pd.read_csv("./data/application_train_vf.csv")

    for var in train.columns:
        if train[var].nunique() < 30:
            train[var] = train[var].astype("object")
    print("Type des variables convertis ✅")

    impute_0 = ["OWN_CAR_AGE", "YEARS_BEGINEXPLUATATION_MEDI","YEARS_BEGINEXPLUATATION_MODE",
                "YEARS_BEGINEXPLUATATION_AVG" ]

    for var in impute_0:
        train[var].fillna(0, inplace=True)

    #### EXT_SOURCE ####
    train['EXT_SOURCE_1'].fillna( train['EXT_SOURCE_2'], inplace=True)
    train['EXT_SOURCE_1'].fillna( train["EXT_SOURCE_1"].mean(), inplace=True)
    train['EXT_SOURCE_3'].fillna( train['EXT_SOURCE_2'], inplace=True)
    train['EXT_SOURCE_3'].fillna( train["EXT_SOURCE_3"].mean(), inplace=True)

    ### Others ####
    impute_mod = ["OCCUPATION_TYPE"]
    for var in impute_mod:
         train[var].fillna( train[var].mode()[0], inplace=True)

    #### EXT_SOURCE ####
    train['EXT_SOURCE_1'].fillna( train['EXT_SOURCE_2'], inplace=True)
    train['EXT_SOURCE_1'].fillna( train["EXT_SOURCE_1"].mean(), inplace=True)
    train['EXT_SOURCE_3'].fillna( train['EXT_SOURCE_2'], inplace=True)
    train['EXT_SOURCE_3'].fillna( train["EXT_SOURCE_3"].mean(), inplace=True)

    for var in  train.columns:
        pcentage_nan =  train[var].isna().sum() /  train.shape[0]

        if pcentage_nan != 0:
            if pcentage_nan > 0.3:
                 train.drop(columns=[var], inplace=True)
            else:
                if train[var].dtype != 'object':
                    mean =  train[var].mean()
                    train[var].fillna(mean, inplace=True)
                else :
                    mode =  train[var].mode()
                    train[var].fillna(mode[0], inplace=True)

    dict_variable = {
        "EXT_SOURCE_2" : 3,
        "EXT_SOURCE_1" : 2,
        "EXT_SOURCE_3" :2
    }

    start_time = time.time()
    discretizer = Genetic_Numerical_Discretisation(train, dict_variable, False)
    t = discretizer.run_discretisation()
    end_time = time.time()

    print(f"Temps passé : {end_time - start_time}")

    print(t["EXT_SOURCE_2_disc_int"])
    print(t["EXT_SOURCE_3_disc_int"])
    print(t["EXT_SOURCE_1_disc_int"])


