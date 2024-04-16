import pandas as pd
from data_preparation import *
from modelization import *
from analyse import *

dataprep = DashDataPreparation()
model_challenger = Modelization()
model_classique = Modelization()
analyse = Analyse()

dictionnaire = pd.read_csv("./datas/Columns_Description.csv")
