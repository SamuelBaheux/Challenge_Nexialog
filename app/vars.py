import pandas as pd
from data_preparation import *
from modelization import *

dataprep = DashDataPreparation()
model = Modelization()
statement_list = ["test"]

df = pd.read_csv("./datas/df_score.csv")
dictionnaire = pd.read_csv("./datas/Columns_Description.csv")
