from Logit_utils import *
from XGB_utils import *

class Modelization():
    def init_model(self, model):
        if model == 'logit':
            self.model = LogitModel()
            self.model_name = "logit"
        else :
            self.model = XGB_model()
            self.model_name = "xgb"

    def init_data(self, data, intervalles_dic):
        self.model.init_data(data, intervalles_dic)

    def run_model(self):
        self.results = self.model.run_model()

    def get_metrics(self):
        return(self.model.get_metrics())

    def get_grid_score(self, train_prepared):
        if self.model_name == 'logit':
            GS = GridScore(train_prepared, self.results)
            grid_score = GS.compute_grid_score()
            return(grid_score)

        else :
            gs = GridScoreXGB(train_prepared, self.results)
            grid_score = gs.compute_grid_score()
            return(grid_score)

