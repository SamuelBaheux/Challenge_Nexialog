import warnings
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from scipy.stats import chi2_contingency
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning, module="deap.creator")
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


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

    def run_discretisation(self):
        self.train["date_mensuelle"] = pd.to_datetime(self.train["date_mensuelle"])
        self.train['date_trimestrielle'] = (self.train['date_mensuelle'].dt.year.astype(str) + '_' +
                                            self.train['date_mensuelle'].dt.quarter.astype(str))

        self.intervalles_dic = {}

        for variable in tqdm(self.variables_dict):
            bins = self.genetic_discretisation(self.train, variable, self.variables_dict[variable] - 1)
            bins_normalise = sorted([min(1, max(0, val)) for val in bins])

            seuils = np.percentile(self.train[variable], [val * 100 for val in bins_normalise])
            seuils_uniques = np.unique(seuils)

            self.train[f'{variable}_disc'] = np.digitize(self.train[variable], seuils_uniques)

            intervalles = self.train.groupby(f'{variable}_disc')[variable].agg(['min', 'max'])

            dict_renommage = {modalite: f'[{round(row["min"], 2)};{round(row["max"], 2)}]' for modalite, row in
                              intervalles.iterrows()}

            self.intervalles_dic[variable] = dict_renommage

            self.train[f'{variable}_disc_int'] = self.train[f'{variable}_disc'].map(dict_renommage)

            if self.plot:
                self.plot_stability(f'{variable}_disc')

        return (self.train)


class DataPreparation():
    def __init__(self):
        self.train = pd.read_csv('../../data/application_train_vf.csv')
        self.nan_treshold = 0.3
        self.plot = False

    def add_external_features(self):
        df_bur = pd.read_csv('../../data/bureau.csv')
        df_bur_group = df_bur[['DAYS_CREDIT_ENDDATE', "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", 'SK_ID_CURR']].groupby('SK_ID_CURR').sum()
        df_bur_group.reset_index(inplace=True)

        df_prev = pd.read_csv('../../data/previous_application.csv')
        df_prev_group = df_prev[['SK_ID_CURR', 'DAYS_FIRST_DRAWING', 'RATE_DOWN_PAYMENT']].groupby(
            'SK_ID_CURR').sum()
        df_prev_group.reset_index(inplace=True)

        df_ins = pd.read_csv('../../data/installments_payments.csv')
        df_ins_group = df_ins[['SK_ID_CURR', 'AMT_PAYMENT']].groupby('SK_ID_CURR').sum()
        df_ins_group.reset_index(inplace=True)

        self.train = self.train.merge(df_bur_group, on='SK_ID_CURR', how='left')
        self.train = self.train.merge(df_prev_group, on='SK_ID_CURR', how='left')
        self.train = self.train.merge(df_ins_group, on='SK_ID_CURR', how='left')
        print("Variables extérieures récupérées ✅")

    def convert_type(self):
        for var in self.train.columns:
            if self.train[var].nunique() < 30:
                self.train[var] = self.train[var].astype("object")
        print("Type des variables convertis ✅")

    def remove_and_impute_nan(self):
        ### Exceptions ####
        impute_0 = ["OWN_CAR_AGE", "YEARS_BEGINEXPLUATATION_MEDI","YEARS_BEGINEXPLUATATION_MODE",
                    "YEARS_BEGINEXPLUATATION_AVG", "DAYS_CREDIT_ENDDATE",
                    "DAYS_FIRST_DRAWING", "RATE_DOWN_PAYMENT", "AMT_PAYMENT"]

        for var in impute_0:
            self.train[var].fillna(0, inplace=True)

        for var in ['AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM']:
            self.train[var].fillna(self.train[var].median(), inplace=True)

        #### EXT_SOURCE ####
        self.train['EXT_SOURCE_1'].fillna(self.train['EXT_SOURCE_2'], inplace=True)
        self.train['EXT_SOURCE_1'].fillna(self.train["EXT_SOURCE_1"].mean(), inplace=True)
        self.train['EXT_SOURCE_3'].fillna(self.train['EXT_SOURCE_2'], inplace=True)
        self.train['EXT_SOURCE_3'].fillna(self.train["EXT_SOURCE_3"].mean(), inplace=True)

        ### Others ####
        impute_mod = ["OCCUPATION_TYPE"]
        for var in impute_mod:
            self.train[var].fillna(self.train[var].mode()[0], inplace=True)

        #### EXT_SOURCE ####
        self.train['EXT_SOURCE_1'].fillna(self.train['EXT_SOURCE_2'], inplace=True)
        self.train['EXT_SOURCE_1'].fillna(self.train["EXT_SOURCE_1"].mean(), inplace=True)
        self.train['EXT_SOURCE_3'].fillna(self.train['EXT_SOURCE_2'], inplace=True)
        self.train['EXT_SOURCE_3'].fillna(self.train["EXT_SOURCE_3"].mean(), inplace=True)

        for var in self.train.columns:
            pcentage_nan = self.train[var].isna().sum() / self.train.shape[0]

            if pcentage_nan != 0:
                if pcentage_nan > self.nan_treshold:
                    self.train.drop(columns=[var], inplace=True)
                else:
                    if self.train[var].dtype != 'object':
                        mean = self.train[var].mean()
                        self.train[var].fillna(mean, inplace=True)
                    else:
                        mode = self.train[var].mode()
                        self.train[var].fillna(mode[0], inplace=True)

        assert (self.train.isna().sum().sum() == 0)
        print("Valeurs manquantes traitées ✅")

    def numericals_discretisation(self):
        print("Discrétisation des variables numériques en cours ... ")
        var_3_bins = ["EXT_SOURCE_2", "EXT_SOURCE_1"]

        #var_2_bins = ["AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "AMT_GOODS_PRICE", "DAYS_REGISTRATION", "DAYS_LAST_PHONE_CHANGE", "EXT_SOURCE_3",
                      #"AMT_CREDIT", "AMT_ANNUITY", "REGION_POPULATION_RELATIVE", "DAYS_EMPLOYED",
                      #"DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "AMT_REQ_CREDIT_BUREAU_MON",
                      #"OWN_CAR_AGE", "YEARS_BEGINEXPLUATATION_MEDI",
                      #"YEARS_BEGINEXPLUATATION_MODE", "YEARS_BEGINEXPLUATATION_AVG", 'REGION_RATING_CLIENT_W_CITY',
                      #'EXT_SOURCE_2', 'DAYS_CREDIT_ENDDATE', 'CNT_PAYMENT',
                      #'DAYS_FIRST_DRAWING', 'RATE_DOWN_PAYMENT', 'AMT_PAYMENT']

        var_2_bins = ['DAYS_CREDIT_ENDDATE', 'DAYS_FIRST_DRAWING', 'RATE_DOWN_PAYMENT', 'AMT_PAYMENT',
                      "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "DAYS_EMPLOYED", "EXT_SOURCE_3"]

        dict_variable = {}

        for var in var_2_bins:
            dict_variable[var] = 2

        for var in var_3_bins:
            dict_variable[var] = 3

        self.discretizer = Genetic_Numerical_Discretisation(self.train, dict_variable, self.plot)
        self.train = self.discretizer.run_discretisation()

        print("Variables numériques discrétisées ✅")

    def categorical_discretisation(self):
        print("Discrétisation des variables catégorielles en cours ... ")
        #### NAME INCOME TYPE ####
        high_income = ["Working", "Commercial associate", "Businessman"]
        other = ['State servant', 'Pensioner', 'Student', 'Maternity leave', 'Unemployed']

        self.train['NAME_INCOME_TYPE_discret'] = np.select([self.train['NAME_INCOME_TYPE'].isin(high_income),
                                                            self.train['NAME_INCOME_TYPE'].isin(other)],
                                                           ['high_income', 'Low_income'],
                                                           default='Low_income')

        #### NAME EDUCATION TYPE ####
        lower = ["Lower_education", "Secondary / secondary special", "Incomplete higher"]
        higher = ["Higher education", "Academic degree"]

        self.train['NAME_EDUCATION_TYPE_discret'] = np.select([self.train['NAME_EDUCATION_TYPE'].isin(lower),
                                                               self.train['NAME_EDUCATION_TYPE'].isin(higher)],
                                                              ['lower', 'higher'],
                                                              default='lower')


        #### NAME FAMILY STATUS ###

        alone = ["Single / not married", "Separated", "Widow", "Security staff", "Laborers", "Unknown",
                 "Civil marriage"]
        couple = ["Married"]

        self.train['NAME_FAMILY_STATUS_discret'] = np.select([self.train['NAME_FAMILY_STATUS'].isin(alone),
                                                              self.train['NAME_FAMILY_STATUS'].isin(couple)],
                                                             ['alone', 'couple'],
                                                             default='couple')


        #### OCCUPATION TYPE ###

        low_skilled = ["Low-skill Laborers", "Drivers", "Waiters/barmen staff", "Security staff", "Laborers",
                       "Sales staff", "Cooking staff", "Cleaning staff", "Realty agents", "Secretaries"]
        high_skilled = ["Medicine staff", "IT staff", "Private service staff", "Managers", "Core staff", "HR staff",
                        "Accountants", "High skilled tech staff"]

        self.train['OCCUPATION_TYPE_discret'] = np.select([self.train['OCCUPATION_TYPE'].isin(low_skilled),
                                                           self.train['OCCUPATION_TYPE'].isin(high_skilled)],
                                                          ['low_skilled', 'high_skilled'],
                                                          default='low_skilled')


        #### CODE GENDER ####
        mode_gender = self.train["CODE_GENDER"].mode()[0]
        self.train['CODE_GENDER'].replace('XNA', mode_gender, inplace=True)

        ### REGION_RATING_CLIENT_W_CITY ###
        self.train['REGION_RATING_CLIENT_W_CITY'].replace(1, 2, inplace=True)
        self.train['REGION_RATING_CLIENT_W_CITY'].replace(2, "un_deux", inplace=True)

        ### NAME_CONTRACT_TYPE ###
        self.train["NAME_CONTRACT_TYPE"] = self.train["NAME_CONTRACT_TYPE"].str.replace(' ', '_')

        print("Variables catégorielles discrétisées ✅")

    def rename_categories(self):
        var_num_to_str = ['FLAG_EMP_PHONE', 'REG_CITY_NOT_LIVE_CITY',
                          'REG_CITY_NOT_WORK_CITY', 'REGION_RATING_CLIENT',
                          'REGION_RATING_CLIENT_W_CITY', 'FLAG_WORK_PHONE',
                          'FLAG_PHONE', 'LIVE_CITY_NOT_WORK_CITY']

        replacement_dict = {1: 'un', 0: 'zero', 2: 'deux', 3: 'trois'}

        for var in var_num_to_str :
            self.train[var] = self.train[var].replace(replacement_dict)

        self.train["TARGET"] = self.train["TARGET"].astype("int")

    def get_prepared_data(self):
        self.add_external_features()
        self.convert_type()
        self.remove_and_impute_nan()
        self.numericals_discretisation()
        self.categorical_discretisation()
        self.rename_categories()

        numericals = [var for var in self.train.columns if '_disc_int' in var]
        categoricals = [var for var in self.train.columns if '_discret' in var]
        already_prepared = ['FLAG_EMP_PHONE', 'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY',
                            'REGION_RATING_CLIENT','REGION_RATING_CLIENT_W_CITY', "FLAG_WORK_PHONE",
                            "FLAG_PHONE", "LIVE_CITY_NOT_WORK_CITY",'NAME_CONTRACT_TYPE', 'FLAG_OWN_CAR',
                            'FLAG_OWN_REALTY', 'CODE_GENDER']

        other = ["date_mensuelle", "TARGET"]

        final_features = other + numericals + categoricals + already_prepared

        return (self.train[final_features])

class DashDataPreparation():
    def __init__(self):
        self.train = pd.read_csv('./data/application_train_vf.csv')
        self.nan_treshold = 0.3
        self.plot = False

    def add_external_features(self):
        df_bur = pd.read_csv('./data/bureau.csv')
        df_bur_group = df_bur[['DAYS_CREDIT_ENDDATE', "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", 'SK_ID_CURR']].groupby('SK_ID_CURR').sum()
        df_bur_group.reset_index(inplace=True)

        df_prev = pd.read_csv('./data/previous_application.csv')
        df_prev_group = df_prev[['SK_ID_CURR', 'DAYS_FIRST_DRAWING', 'RATE_DOWN_PAYMENT']].groupby(
            'SK_ID_CURR').sum()
        df_prev_group.reset_index(inplace=True)

        df_ins = pd.read_csv('./data/installments_payments.csv')
        df_ins_group = df_ins[['SK_ID_CURR', 'AMT_PAYMENT']].groupby('SK_ID_CURR').sum()
        df_ins_group.reset_index(inplace=True)

        self.train = self.train.merge(df_bur_group, on='SK_ID_CURR', how='left')
        self.train = self.train.merge(df_prev_group, on='SK_ID_CURR', how='left')
        self.train = self.train.merge(df_ins_group, on='SK_ID_CURR', how='left')
        print("Variables extérieures récupérées ✅")

    def convert_type(self):
        for var in self.train.columns:
            if self.train[var].nunique() < 30:
                self.train[var] = self.train[var].astype("object")
        print("Type des variables convertis ✅")

    def remove_and_impute_nan(self):
        ### Exceptions ####
        impute_0 = ["OWN_CAR_AGE", "YEARS_BEGINEXPLUATATION_MEDI","YEARS_BEGINEXPLUATATION_MODE",
                    "YEARS_BEGINEXPLUATATION_AVG", "DAYS_CREDIT_ENDDATE",
                    "DAYS_FIRST_DRAWING", "RATE_DOWN_PAYMENT", "AMT_PAYMENT"]

        for var in impute_0:
            self.train[var].fillna(0, inplace=True)

        for var in ['AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM']:
            self.train[var].fillna(self.train[var].median(), inplace=True)

        #### EXT_SOURCE ####
        self.train['EXT_SOURCE_1'].fillna(self.train['EXT_SOURCE_2'], inplace=True)
        self.train['EXT_SOURCE_1'].fillna(self.train["EXT_SOURCE_1"].mean(), inplace=True)
        self.train['EXT_SOURCE_3'].fillna(self.train['EXT_SOURCE_2'], inplace=True)
        self.train['EXT_SOURCE_3'].fillna(self.train["EXT_SOURCE_3"].mean(), inplace=True)

        ### Others ####
        impute_mod = ["OCCUPATION_TYPE"]
        for var in impute_mod:
            self.train[var].fillna(self.train[var].mode()[0], inplace=True)

        #### EXT_SOURCE ####
        self.train['EXT_SOURCE_1'].fillna(self.train['EXT_SOURCE_2'], inplace=True)
        self.train['EXT_SOURCE_1'].fillna(self.train["EXT_SOURCE_1"].mean(), inplace=True)
        self.train['EXT_SOURCE_3'].fillna(self.train['EXT_SOURCE_2'], inplace=True)
        self.train['EXT_SOURCE_3'].fillna(self.train["EXT_SOURCE_3"].mean(), inplace=True)

        for var in self.train.columns:
            pcentage_nan = self.train[var].isna().sum() / self.train.shape[0]

            if pcentage_nan != 0:
                if pcentage_nan > self.nan_treshold:
                    self.train.drop(columns=[var], inplace=True)
                else:
                    if self.train[var].dtype != 'object':
                        mean = self.train[var].mean()
                        self.train[var].fillna(mean, inplace=True)
                    else:
                        mode = self.train[var].mode()
                        self.train[var].fillna(mode[0], inplace=True)

        assert (self.train.isna().sum().sum() == 0)
        print("Valeurs manquantes traitées ✅")

    def intialize_data(self):
        self.add_external_features()
        self.convert_type()
        self.remove_and_impute_nan()

    def get_features(self):
        return(self.train.columns.to_list())

    def numericals_discretisation(self):
        print("Discrétisation des variables numériques en cours ... ")
        var_3_bins = ["EXT_SOURCE_2", "EXT_SOURCE_1"]

        #var_2_bins = ["AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "AMT_GOODS_PRICE", "DAYS_REGISTRATION", "DAYS_LAST_PHONE_CHANGE", "EXT_SOURCE_3",
                      #"AMT_CREDIT", "AMT_ANNUITY", "REGION_POPULATION_RELATIVE", "DAYS_EMPLOYED",
                      #"DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "AMT_REQ_CREDIT_BUREAU_MON",
                      #"OWN_CAR_AGE", "YEARS_BEGINEXPLUATATION_MEDI",
                      #"YEARS_BEGINEXPLUATATION_MODE", "YEARS_BEGINEXPLUATATION_AVG", 'REGION_RATING_CLIENT_W_CITY',
                      #'EXT_SOURCE_2', 'DAYS_CREDIT_ENDDATE', 'CNT_PAYMENT',
                      #'DAYS_FIRST_DRAWING', 'RATE_DOWN_PAYMENT', 'AMT_PAYMENT']

        var_2_bins = ['DAYS_CREDIT_ENDDATE', 'DAYS_FIRST_DRAWING', 'RATE_DOWN_PAYMENT', 'AMT_PAYMENT',
                      "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "DAYS_EMPLOYED", "EXT_SOURCE_3"]

        dict_variable = {}

        for var in var_2_bins:
            dict_variable[var] = 2

        for var in var_3_bins:
            dict_variable[var] = 3

        self.discretizer = Genetic_Numerical_Discretisation(self.train, dict_variable, self.plot)
        self.train = self.discretizer.run_discretisation()

        print("Variables numériques discrétisées ✅")

    def categorical_discretisation(self):
        print("Discrétisation des variables catégorielles en cours ... ")
        #### NAME INCOME TYPE ####
        high_income = ["Working", "Commercial associate", "Businessman"]
        other = ['State servant', 'Pensioner', 'Student', 'Maternity leave', 'Unemployed']

        self.train['NAME_INCOME_TYPE_discret'] = np.select([self.train['NAME_INCOME_TYPE'].isin(high_income),
                                                            self.train['NAME_INCOME_TYPE'].isin(other)],
                                                           ['high_income', 'Low_income'],
                                                           default='Low_income')

        #### NAME EDUCATION TYPE ####
        lower = ["Lower_education", "Secondary / secondary special", "Incomplete higher"]
        higher = ["Higher education", "Academic degree"]

        self.train['NAME_EDUCATION_TYPE_discret'] = np.select([self.train['NAME_EDUCATION_TYPE'].isin(lower),
                                                               self.train['NAME_EDUCATION_TYPE'].isin(higher)],
                                                              ['lower', 'higher'],
                                                              default='lower')


        #### NAME FAMILY STATUS ###

        alone = ["Single / not married", "Separated", "Widow", "Security staff", "Laborers", "Unknown",
                 "Civil marriage"]
        couple = ["Married"]

        self.train['NAME_FAMILY_STATUS_discret'] = np.select([self.train['NAME_FAMILY_STATUS'].isin(alone),
                                                              self.train['NAME_FAMILY_STATUS'].isin(couple)],
                                                             ['alone', 'couple'],
                                                             default='couple')


        #### OCCUPATION TYPE ###

        low_skilled = ["Low-skill Laborers", "Drivers", "Waiters/barmen staff", "Security staff", "Laborers",
                       "Sales staff", "Cooking staff", "Cleaning staff", "Realty agents", "Secretaries"]
        high_skilled = ["Medicine staff", "IT staff", "Private service staff", "Managers", "Core staff", "HR staff",
                        "Accountants", "High skilled tech staff"]

        self.train['OCCUPATION_TYPE_discret'] = np.select([self.train['OCCUPATION_TYPE'].isin(low_skilled),
                                                           self.train['OCCUPATION_TYPE'].isin(high_skilled)],
                                                          ['low_skilled', 'high_skilled'],
                                                          default='low_skilled')


        #### CODE GENDER ####
        mode_gender = self.train["CODE_GENDER"].mode()[0]
        self.train['CODE_GENDER'].replace('XNA', mode_gender, inplace=True)

        ### REGION_RATING_CLIENT_W_CITY ###
        self.train['REGION_RATING_CLIENT_W_CITY'].replace(1, 2, inplace=True)
        self.train['REGION_RATING_CLIENT_W_CITY'].replace(2, "un_deux", inplace=True)

        ### NAME_CONTRACT_TYPE ###
        self.train["NAME_CONTRACT_TYPE"] = self.train["NAME_CONTRACT_TYPE"].str.replace(' ', '_')

        print("Variables catégorielles discrétisées ✅")

    def rename_categories(self):
        var_num_to_str = ['FLAG_EMP_PHONE', 'REG_CITY_NOT_LIVE_CITY',
                          'REG_CITY_NOT_WORK_CITY', 'REGION_RATING_CLIENT',
                          'REGION_RATING_CLIENT_W_CITY', 'FLAG_WORK_PHONE',
                          'FLAG_PHONE', 'LIVE_CITY_NOT_WORK_CITY']

        replacement_dict = {1: 'un', 0: 'zero', 2: 'deux', 3: 'trois'}

        for var in var_num_to_str :
            self.train[var] = self.train[var].replace(replacement_dict)

        self.train["TARGET"] = self.train["TARGET"].astype("int")

    def get_prepared_data(self):
        self.numericals_discretisation()
        self.categorical_discretisation()
        self.rename_categories()

        numericals = [var for var in self.train.columns if '_disc_int' in var]
        categoricals = [var for var in self.train.columns if '_discret' in var]
        already_prepared = ['FLAG_EMP_PHONE', 'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY',
                            'REGION_RATING_CLIENT','REGION_RATING_CLIENT_W_CITY', "FLAG_WORK_PHONE",
                            "FLAG_PHONE", "LIVE_CITY_NOT_WORK_CITY",'NAME_CONTRACT_TYPE', 'FLAG_OWN_CAR',
                            'FLAG_OWN_REALTY', 'CODE_GENDER']

        other = ["date_mensuelle", "TARGET"]

        final_features = other + numericals + categoricals + already_prepared

        return (self.train[final_features])
