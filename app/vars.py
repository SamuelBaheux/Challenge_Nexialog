import pandas as pd
from data_preparation import *
from modelization import *
from analyse import *

dataprep = DashDataPreparation()
model_challenger = Modelization()
model_classique = Modelization()

#df_a = pd.read_csv("./data/application_train_vf.csv")
analyse = Analyse()
#analyse.init_data(df_a)
#analyse.init_target("TARGET")
#analyse.init_date("date_mensuelle")


dictionnaire = pd.read_csv("./data/Columns_Description.csv")