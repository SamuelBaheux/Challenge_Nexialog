import pandas as pd
from data_preparation import *
from modelization import *


INFO_VAR_INT = '''Détail des variables choisies pour l\'interprétabilité :
- REGION_RATING_CLIENT_W_CITY : Rating of the region where client lives with taking city into account
- DAYS_CREDIT_ENDDATE : Remaining duration of CB credit (in days) at the time of application in Home Credit
- RATE_DOWN_PAYMENT : Down payment rate normalized on previous credit
- AMT_PAYMENT : What the client actually paid on previous credit on this installment
- NAME_INCOME_TYPE : Clients income type
- OCCUPATION_TYPE : What kind of occupation does the client have
'''

INFO_VAR_PERF =  '''Détail des variables choisies pour la performance :
- AMT_CREDIT_SUM_DEBT : Current debt on Credit Bureau credit
- AMT_CREDIT_SUM : Current credit amount for the Credit Bureau credit
- EXT_SOURCE_1 : Normalized score from external data source 1
- EXT_SOURCE_2 : Normalized score from external data source 2
- EXT_SOURCE_3 : Normalized score from external data source 3
- NAME_INCOME_TYPE : Clients income type
- DAYS_EMPLOYED : How many days before the application the person started current employment
'''

dataprep = DashDataPreparation()
model = Modelization()

df = pd.read_csv("./data/df_score.csv")