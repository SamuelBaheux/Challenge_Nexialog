class Analyse():
    def __init__(self):
        self.df = None

    def init_data(self, df):
        self.df = df

    def get_features(self):
        if self.df is not None:
            return(self.df.columns.to_list())
        else :
            return []