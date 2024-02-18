import pandas as pd
import numpy as np
from deap import base, creator, tools, algorithms
from functools import  partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore", category=RuntimeWarning, module="deap.creator")


class Genetic_Numerical_Discretisation():
    def __init__(self, train, test, variables_dict, plot = False):
        self.train = train
        self.test = test
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
        stability_df = self.train.groupby(['date_trimestrielle', variable])['TARGET'].mean().unstack()
        stability_df['stability'] = stability_df.std(axis=1) / stability_df.mean(axis=1)

        plt.figure(figsize=(10, 5))

        for class_label in stability_df.drop('stability', axis=1).columns:
            values = stability_df[class_label]
            plt.plot(stability_df.index, values, label=f'Classe {class_label}', marker='o')

        plt.title(f'Stabilité de l\'impact sur la cible pour {variable}')
        plt.xlabel('Date')
        plt.ylabel('Proportion de la cible TARGET')
        plt.legend(title=f'Classes de_binned', loc='upper left', bbox_to_anchor=(1, 1))
        plt.show()

    def genetic_discretisation(self, train_set, variable, nb_classes_max):
        temp = train_set[[variable, 'TARGET']].copy()
        NB_GEN = 15
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

    def run_discretisation(self):
        self.train["date_mensuelle"] = pd.to_datetime(self.train["date_mensuelle"])
        self.train['date_trimestrielle'] = (self.train['date_mensuelle'].dt.year.astype(str) + '_' +
                                            self.train['date_mensuelle'].dt.quarter.astype(str))

        for variable in tqdm(self.variables_dict):
            bins = self.genetic_discretisation(self.train, variable, self.variables_dict[variable] - 1)
            bins_normalise = sorted([min(1, max(0, val)) for val in bins])

            seuils = np.percentile(self.train[variable], [val * 100 for val in bins_normalise])
            seuils_uniques = np.unique(seuils)

            self.train[f'{variable}_disc'] = np.digitize(self.train[variable], seuils_uniques)
            self.test[f'{variable}_disc'] = np.digitize(self.test[variable], seuils_uniques)

            intervalles = self.train.groupby(f'{variable}_disc')[variable].agg(['min', 'max'])

            dict_renommage = {modalite: f'[{round(row["min"], 2)}-{round(row["max"], 2)}]' for modalite, row in
                              intervalles.iterrows()}

            self.train[f'{variable}_disc_int'] = self.train[f'{variable}_disc'].map(dict_renommage)
            self.test[f'{variable}_disc_int'] = self.test[f'{variable}_disc'].map(dict_renommage)

            if self.plot:
                self.plot_stability(f'{variable}_disc')

        return (self.train, self.test)



class DataPreparation():
    def __init__(self, train, test, nan_treshold) :
        self.train = train
        self.test = test
        self.nan_treshold = nan_treshold

    def convert_type(self):
        for var in self.train.columns :
            if self.train[var].nunique() < 30 :
                self.train[var] = self.train[var].astype("object")
                if var != "TARGET" :
                    self.test[var] = self.test[var].astype("object")
        print("Type des variables convertis ✅")

    def remove_and_impute_nan(self):
        for var in self.train.columns :
            pcentage_nan = self.train[var].isna().sum()/self.train.shape[0]

            if pcentage_nan != 0 :
                if pcentage_nan > self.nan_treshold :
                    self.train.drop(columns = [var], inplace = True)
                    self.test.drop(columns = [var], inplace=True)
                else :
                    if self.train[var].dtype != 'object' :
                        mean = self.train[var].mean()
                        self.train[var].fillna(mean, inplace =True)
                        self.test[var].fillna(mean, inplace=True)
                    else :
                        mode = self.train[var].mode()
                        self.train[var].fillna(mode[0], inplace=True)
                        self.test[var].fillna(mode[0], inplace=True)

        assert (self.train.isna().sum().sum() == 0) and (self.test.isna().sum().sum() == 0)
        print("Valeurs manquantes traitées ✅")

    def numericals_discretisation(self):
        print("Discrétisation des variables numériques en cours ... ")
        var_3_bins = ["DAYS_BIRTH", "EXT_SOURCE_2"]

        var_2_bins = ["AMT_GOODS_PRICE", "DAYS_REGISTRATION", "DAYS_LAST_PHONE_CHANGE", "EXT_SOURCE_3",
                      "AMT_CREDIT", "AMT_ANNUITY", "REGION_POPULATION_RELATIVE", "DAYS_EMPLOYED",
                      "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "AMT_REQ_CREDIT_BUREAU_MON"]

        dict_variable = {}

        for var in var_3_bins:
            dict_variable[var] = 3

        for var in var_2_bins :
            dict_variable[var] = 2

        discretizer = Genetic_Numerical_Discretisation(self.train, self.test, dict_variable)
        self.train, self.test = discretizer.run_discretisation()

        print("Variables numériques discrétisées ✅")

    def categorical_discretisation(self):
        print("Discrétisation des variables catégorielles en cours ... ")
        #### NAME INCOME TYPE ####
        low_income = ['Maternity leave', 'Unemployed']
        high_income = ["Working", "Commercial associate", "Businessman"]
        other = ['State servant', 'Pensioner', 'Student']

        self.train['NAME_INCOME_TYPE_discret'] = np.select([self.train['NAME_INCOME_TYPE'].isin(low_income),
                                                            self.train['NAME_INCOME_TYPE'].isin(high_income),
                                                            self.train['NAME_INCOME_TYPE'].isin(other)],
                                                           ['low_income', 'high_income', 'other'],
                                                           default='other')

        self.test['NAME_INCOME_TYPE_discret'] = np.select([self.test['NAME_INCOME_TYPE'].isin(low_income),
                                                            self.test['NAME_INCOME_TYPE'].isin(high_income),
                                                            self.test['NAME_INCOME_TYPE'].isin(other)],
                                                           ['low_income', 'high_income', 'other'],
                                                           default='other')

        #### NAME EDUCATION TYPE ####
        lower = ["Lower_education", "Secondary / secondary special", "Incomplete higher"]
        higher = ["Higher education", "Academic degree"]

        self.train['NAME_EDUCATION_TYPE_discret'] = np.select([self.train['NAME_EDUCATION_TYPE'].isin(lower),
                                                               self.train['NAME_EDUCATION_TYPE'].isin(higher)],
                                                              ['lower', 'higher'],
                                                              default='lower')

        self.test['NAME_EDUCATION_TYPE_discret'] = np.select([self.test['NAME_EDUCATION_TYPE'].isin(lower),
                                                               self.test['NAME_EDUCATION_TYPE'].isin(higher)],
                                                              ['lower', 'higher'],
                                                              default='lower')

        #### NAME FAMILY STATUS ###

        alone = ["Single / not married", "Separated", "Widow", "Security staff", "Laborers", "Unknown","Civil marriage"]
        couple = ["Married"]

        self.train['NAME_FAMILY_STATUS_discret'] = np.select([self.train['NAME_FAMILY_STATUS'].isin(alone),
                                                              self.train['NAME_FAMILY_STATUS'].isin(couple)],
                                                             ['alone', 'couple'],
                                                             default='couple')

        self.test['NAME_FAMILY_STATUS_discret'] = np.select([self.test['NAME_FAMILY_STATUS'].isin(alone),
                                                              self.test['NAME_FAMILY_STATUS'].isin(couple)],
                                                             ['alone', 'couple'],
                                                             default='couple')

        print("Variables catégorielles discrétisées ✅")

    def get_prepared_data(self):
        self.convert_type()
        self.remove_and_impute_nan()
        self.numericals_discretisation()
        self.categorical_discretisation()

        numericals = [var for var in self.train.columns if '_disc_int' in var]
        categoricals = [var for var in self.train.columns if '_discret' in var]
        already_prepared = ['FLAG_EMP_PHONE', 'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'REGION_RATING_CLIENT',
                            'REGION_RATING_CLIENT_W_CITY', "FLAG_WORK_PHONE", "FLAG_PHONE", "LIVE_CITY_NOT_WORK_CITY",
                            'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_HOUSING_TYPE']
        other = ["date_mensuelle"]

        final_features_test = other + numericals + categoricals + already_prepared
        final_features_train = ["TARGET"] + final_features_test

        return(self.train[final_features_train], self.test[final_features_test])





