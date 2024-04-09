import pandas as pd
from data_preparation import *
from modelization import *
from analyse import *

dataprep = DashDataPreparation()
model = Modelization()
analyse = Analyse()
statement_list = ["test"]

df = pd.read_csv("./data/df_score.csv")
dictionnaire = pd.read_csv("./data/Columns_Description.csv")