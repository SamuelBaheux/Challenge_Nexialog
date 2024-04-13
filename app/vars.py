import pandas as pd
from data_preparation import *
from modelization import *
from analyse import *

dataprep = DashDataPreparation()
model = Modelization()
analyse = Analyse()
statement_list = ["test"]

dictionnaire = pd.read_csv("./data/Columns_Description.csv")