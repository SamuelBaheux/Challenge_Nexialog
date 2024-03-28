import warnings
from functools import partial
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from scipy.stats import chi2_contingency
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning, module="deap.creator")
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


class Genetic_Numerical_Discretisation():
    def __init__(self, train, variables_dict, target, date, plot=False):
        self.train = train
        self.variables_dict = variables_dict
        self.target = target
        self.date = date
        self.plot = plot

    def evalChi2(self, individual, variable, dataset):
        individual_sorted = sorted(individual)
        percentiles = [max(0, min(i * 100, 100)) for i in individual_sorted]
        thresholds = np.percentile(dataset[variable], percentiles)

        thresholds = np.unique(thresholds)

        disc_var = np.digitize(dataset[variable].dropna(), thresholds)
        dataset['disc_var'] = disc_var

        contingency_table = pd.crosstab(dataset['disc_var'], dataset[self.target])
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

        stability_taux_df = self.train.groupby(['date_trimestrielle', variable])[self.target].mean().unstack()
        stability_taux_df['stability'] = stability_taux_df.std(axis=1) / stability_taux_df.mean(axis=1)

        for class_label in stability_taux_df.drop('stability', axis=1).columns:
            values = stability_taux_df[class_label]
            axes[1].plot(stability_taux_df.index, values, label=f'Classe {class_label}', marker='o')

        axes[1].set_title(f'Stabilité de taux pour {variable}')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Proportion de la cible')
        axes[1].legend(title='Classes de_binned', loc='upper left', bbox_to_anchor=(1, 1))
        axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

    def genetic_discretisation(self, train_set, variable, nb_classes_max):
        temp = train_set[[variable, self.target]].copy()
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
        self.train[self.date] = pd.to_datetime(self.train[self.date])
        self.train['date_trimestrielle'] = (self.train[self.date].dt.year.astype(str) + '_' +
                                            self.train[self.date].dt.quarter.astype(str))

        self.intervalles_dic = {}

        results = Parallel(n_jobs=-1)(delayed(self.discretize_variable)(variable) for variable in self.variables_dict)

        for variable, seuils_uniques, dict_renommage in results:
            self.train[f'{variable}_disc'] = np.digitize(self.train[variable], seuils_uniques)
            self.train[f'{variable}_disc_int'] = self.train[f'{variable}_disc'].map(dict_renommage)
            self.intervalles_dic[variable] = dict_renommage

        return self.train

class DashDataPreparation():
    def __init__(self):
        self.nan_treshold = 0.3
        self.plot = False
        self.train = None
        self.target = None
        self.date = None

    def initialize_df(self, df):
        self.train = df

    def init_target(self, target):
        print(target)
        self.target = target

    def init_date(self, date):
        print(date)
        self.date = date

    def get_features(self):
        if self.train is not None:
            return(self.train.columns.to_list())
        else :
            return []

    def initialize_data(self, selected_vars):
        selected_vars.extend([self.date, self.target])
        if "SK_ID_CURR" in self.train.columns :
            selected_vars.extend(["SK_ID_CURR"])

        self.vars = selected_vars

        self.add_external_features(self.vars)
        self.train = self.train[self.vars]
        self.convert_type()
        self.remove_and_impute_nan()
        self.num_vars = self.train.select_dtypes(exclude='object')
        self.cat_vars = self.train.select_dtypes(include='object')

    def add_external_features(self, selected_features):
        vars_bur = ['DAYS_CREDIT_ENDDATE', "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT"]
        vars_prev = ['DAYS_FIRST_DRAWING', 'RATE_DOWN_PAYMENT']
        vars_ins = ['AMT_PAYMENT']

        vars_bur_selected = [var for var in selected_features if var in vars_bur]
        vars_prev_selected = [var for var in selected_features if var in vars_prev]
        vars_ins_selected = [var for var in selected_features if var in vars_ins]

        if len(vars_bur_selected) > 0 :
            df_bur = pd.read_csv('./data/bureau.csv')
            vars_bur_selected.append('SK_ID_CURR')
            df_bur_group = df_bur[vars_bur_selected].groupby('SK_ID_CURR').sum()
            df_bur_group.reset_index(inplace=True)
            self.train = self.train.merge(df_bur_group, on='SK_ID_CURR', how='left')

        if len(vars_prev_selected) > 0:
            df_prev = pd.read_csv('./data/previous_application.csv')
            vars_prev_selected.append('SK_ID_CURR')
            df_prev_group = df_prev[vars_prev_selected].groupby('SK_ID_CURR').sum()
            df_prev_group.reset_index(inplace=True)
            self.train = self.train.merge(df_prev_group, on='SK_ID_CURR', how='left')

        if len(vars_ins_selected) > 0 :
            df_ins = pd.read_csv('./data/installments_payments.csv')
            df_ins_group = df_ins[['SK_ID_CURR', 'AMT_PAYMENT']].groupby('SK_ID_CURR').sum()
            df_ins_group.reset_index(inplace=True)
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
            if var in self.vars :
                self.train[var].fillna(0, inplace=True)

        for var in ['AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM']:
            if var in self.vars:
                self.train[var].fillna(self.train[var].median(), inplace=True)

        #### EXT_SOURCE ####
        if "EXT_SOURCE_1" in self.vars :
            if 'EXT_SOURCE_2' in self.vars :
                self.train['EXT_SOURCE_1'].fillna(self.train['EXT_SOURCE_2'], inplace=True)
            self.train['EXT_SOURCE_1'].fillna(self.train["EXT_SOURCE_1"].mean(), inplace=True)

        if "EXT_SOURCE_3" in self.vars :
            if 'EXT_SOURCE_2' in self.vars :
                self.train['EXT_SOURCE_3'].fillna(self.train['EXT_SOURCE_2'], inplace=True)
            self.train['EXT_SOURCE_3'].fillna(self.train["EXT_SOURCE_3"].mean(), inplace=True)

        ### Others ####
        impute_mod = ["OCCUPATION_TYPE"]
        for var in impute_mod:
            if var in self.vars:
                self.train[var].fillna(self.train[var].mode()[0], inplace=True)

        for var in self.train.columns:
            if var in self.vars :
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

        var_2_bins = ["AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "AMT_GOODS_PRICE", "DAYS_REGISTRATION",
                      "DAYS_LAST_PHONE_CHANGE", "EXT_SOURCE_3","AMT_CREDIT", "AMT_ANNUITY",
                      "REGION_POPULATION_RELATIVE", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",
                      "AMT_REQ_CREDIT_BUREAU_MON", "OWN_CAR_AGE", "YEARS_BEGINEXPLUATATION_MEDI",
                      "YEARS_BEGINEXPLUATATION_MODE", "YEARS_BEGINEXPLUATATION_AVG", 'REGION_RATING_CLIENT_W_CITY',
                      'EXT_SOURCE_2', 'DAYS_CREDIT_ENDDATE', 'CNT_PAYMENT', 'DAYS_FIRST_DRAWING', 'RATE_DOWN_PAYMENT',
                      'AMT_PAYMENT', 'DAYS_CREDIT_ENDDATE', 'DAYS_FIRST_DRAWING', 'RATE_DOWN_PAYMENT', 'AMT_PAYMENT',
                      "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "DAYS_EMPLOYED", "EXT_SOURCE_3"]

        dict_variable = {}

        for var in var_2_bins:
            if var in self.num_vars :
                dict_variable[var] = 2

        for var in var_3_bins:
            if var in self.num_vars:
                dict_variable[var] = 2

        for var in self.num_vars :
            if var not in dict_variable.keys() and var != "SK_ID_CURR":
                dict_variable[var] = 2

        self.discretizer = Genetic_Numerical_Discretisation(self.train, dict_variable, self.target, self.date, self.plot)
        self.train = self.discretizer.run_discretisation()

        print("Variables numériques discrétisées ✅")

    def categorical_discretisation(self):
        print("Discrétisation des variables catégorielles en cours ... ")
        #### NAME INCOME TYPE ####

        if 'NAME_INCOME_TYPE' in self.cat_vars :
            high_income = ["Working", "Commercial associate", "Businessman"]
            other = ['State servant', 'Pensioner', 'Student', 'Maternity leave', 'Unemployed']

            self.train['NAME_INCOME_TYPE_discret'] = np.select([self.train['NAME_INCOME_TYPE'].isin(high_income),
                                                                self.train['NAME_INCOME_TYPE'].isin(other)],
                                                               ['high_income', 'Low_income'],
                                                               default='Low_income')

        #### NAME EDUCATION TYPE ####
        if 'NAME_EDUCATION_TYPE' in self.cat_vars :
            lower = ["Lower_education", "Secondary / secondary special", "Incomplete higher"]
            higher = ["Higher education", "Academic degree"]

            self.train['NAME_EDUCATION_TYPE_discret'] = np.select([self.train['NAME_EDUCATION_TYPE'].isin(lower),
                                                                   self.train['NAME_EDUCATION_TYPE'].isin(higher)],
                                                                  ['lower', 'higher'],
                                                                  default='lower')


        #### NAME FAMILY STATUS ###
        if 'NAME_FAMILY_STATUS' in self.cat_vars :
            alone = ["Single / not married", "Separated", "Widow", "Security staff", "Laborers", "Unknown",
                     "Civil marriage"]
            couple = ["Married"]

            self.train['NAME_FAMILY_STATUS_discret'] = np.select([self.train['NAME_FAMILY_STATUS'].isin(alone),
                                                                  self.train['NAME_FAMILY_STATUS'].isin(couple)],
                                                                 ['alone', 'couple'],
                                                                 default='couple')


        #### OCCUPATION TYPE ###
        if 'OCCUPATION_TYPE' in self.cat_vars :
            low_skilled = ["Low-skill Laborers", "Drivers", "Waiters/barmen staff", "Security staff", "Laborers",
                           "Sales staff", "Cooking staff", "Cleaning staff", "Realty agents", "Secretaries"]
            high_skilled = ["Medicine staff", "IT staff", "Private service staff", "Managers", "Core staff", "HR staff",
                            "Accountants", "High skilled tech staff"]

            self.train['OCCUPATION_TYPE_discret'] = np.select([self.train['OCCUPATION_TYPE'].isin(low_skilled),
                                                               self.train['OCCUPATION_TYPE'].isin(high_skilled)],
                                                              ['low_skilled', 'high_skilled'],
                                                              default='low_skilled')


        #### CODE GENDER ####
        if 'CODE_GENDER' in self.cat_vars :
            mode_gender = self.train["CODE_GENDER"].mode()[0]
            self.train['CODE_GENDER'].replace('XNA', mode_gender, inplace=True)

        ### REGION_RATING_CLIENT_W_CITY ###
        if 'REGION_RATING_CLIENT_W_CITY' in self.cat_vars :
            self.train['REGION_RATING_CLIENT_W_CITY'].replace(1, 2, inplace=True)
            self.train['REGION_RATING_CLIENT_W_CITY'].replace(2, "un_deux", inplace=True)

        ### NAME_CONTRACT_TYPE ###
        if 'NAME_CONTRACT_TYPE' in self.cat_vars :
            self.train["NAME_CONTRACT_TYPE"] = self.train["NAME_CONTRACT_TYPE"].str.replace(' ', '_')

        print("Variables catégorielles discrétisées ✅")

    def rename_categories(self):
        var_num_to_str = ['FLAG_EMP_PHONE', 'REG_CITY_NOT_LIVE_CITY',
                          'REG_CITY_NOT_WORK_CITY', 'REGION_RATING_CLIENT',
                          'REGION_RATING_CLIENT_W_CITY', 'FLAG_WORK_PHONE',
                          'FLAG_PHONE', 'LIVE_CITY_NOT_WORK_CITY']

        replacement_dict = {1: 'un', 0: 'zero', 2: 'deux', 3: 'trois'}

        for var in var_num_to_str :
            if var in self.train.columns :
                self.train[var] = self.train[var].replace(replacement_dict)

        self.train[self.target] = self.train[self.target].astype("int")

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

        already_prepared_bis = [var for var in already_prepared if var in self.train.columns]

        other = [self.date, self.target]

        final_features = other + numericals + categoricals + already_prepared_bis
        self.train = self.train[final_features]
        return (self.train)

    def get_explicative_features(self):
        features = self.train.columns.to_list()
        features.remove(self.target)
        features.remove(self.date)

        features_label = [var.split('_disc_int')[0] for var in features]
        features_label = [var.split('_discret')[0] for var in features_label]
        return(features, features_label)


class ConstantFeatures():
    def __init__(self):

        self.dic_ref = {'AMT_GOODS_PRICE_disc_int': 'max', 'DAYS_LAST_PHONE_CHANGE_disc_int': 'min',
                        'AMT_CREDIT_disc_int': 'min','AMT_ANNUITY_disc_int': 'min',
                        'REGION_POPULATION_RELATIVE_disc_int': 'max','DAYS_ID_PUBLISH_disc_int': 'min',
                        'AMT_REQ_CREDIT_BUREAU_MON_disc_int': 'max','YEARS_BEGINEXPLUATATION_MEDI_disc_int': 'max',
                        'YEARS_BEGINEXPLUATATION_MODE_disc_int': 'max','YEARS_BEGINEXPLUATATION_AVG_disc_int': 'max',
                        'DAYS_FIRST_DRAWING_disc_int': 'max','RATE_DOWN_PAYMENT_disc_int': 'max',
                        'AMT_PAYMENT_disc_int': 'max','AMT_CREDIT_SUM_disc_int': 'max','EXT_SOURCE_2_disc_int': 'max',
                        'EXT_SOURCE_1_disc_int': 'max','EXT_SOURCE_3_disc_int': 'max',
                        'NAME_EDUCATION_TYPE_discret': 'higher', 'NAME_FAMILY_STATUS_discret': 'couple',
                        'FLAG_EMP_PHONE': 'zero', 'REG_CITY_NOT_LIVE_CITY': 'zero', 'REG_CITY_NOT_WORK_CITY': 'un',
                        'REGION_RATING_CLIENT': 'un', 'FLAG_WORK_PHONE': 'zero', 'FLAG_PHONE': 'un',
                        'LIVE_CITY_NOT_WORK_CITY': 'zero', 'FLAG_OWN_CAR': 'Y','FLAG_OWN_REALTY': 'N',
                        'CODE_GENDER': 'F',     'REGION_RATING_CLIENT_W_CITY': "un_deux",
                        'OCCUPATION_TYPE_discret': "high_skilled",'NAME_CONTRACT_TYPE': "Revolving_loans",
                        'NAME_INCOME_TYPE_discret': "Low_income", 'DAYS_CREDIT_ENDDATE_disc_int' : 'min',
                        'AMT_CREDIT_SUM_DEBT_disc_int': 'min','DAYS_EMPLOYED_disc_int': 'min'}