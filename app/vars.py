import pandas as pd
from data_preparation import *
from modelization import *

dataprep = DashDataPreparation()
model = Modelization()

df = pd.read_csv("./data/df_score.csv")
dictionnaire = pd.read_csv("./data/Columns_Description.csv")