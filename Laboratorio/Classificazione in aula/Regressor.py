from sklearn.linear_model    import LinearRegression as LR
from sklearn.svm             import SVR
from sklearn.neighbors       import KNeighborsRegressor   as KNR
from sklearn.tree            import DecisionTreeRegressor as DTR
from sklearn.ensemble        import RandomForestRegressor as RFR
from sklearn.model_selection import GridSearchCV

regressori = {
    "lr" : LR,
    "svr": SVR,
    "knr": KNR,
    "dtr": DTR,
    "rfr": RFR
}

regressori_nomi = {
    "lr": "LinearRegression",
    "svr": "SVR",
    "knr": "KNeighborsRegressor",
    "dtr": "DecisionTreeRegressor",
    "rfr": "RandomForestRegressor"
}

class Regressor:
    def __init__(self, regressor_code):
        if regressor_code not in regressori:
            print("Il regressore non è presente in lista")
            exit(1)
        self.regressor_code = regressor_code
        self.regressor_name = regressori_nomi[regressor_code]
        self.best_score   = None
        self.best_params  = None
        self.final_model  = None
        
    def best_parameters_computing(self, param_grid, scoring, data_obj, cv=5, feat_mtx=None):
        model_template = regressori[self.regressor_code]()
        clf = GridSearchCV(model_template, param_grid, scoring=scoring, cv=cv)
        clf.fit(data_obj.ftx_mtx_tr if feat_mtx is None else feat_mtx, data_obj.targ_vt_tr)
        self.best_score  = clf.best_score_
        self.best_params = clf.best_params_
        self.final_model = regressori[self.regressor_code](**self.best_params)
        self.final_model.fit(data_obj.ftx_mtx_tr if feat_mtx is None else feat_mtx, data_obj.targ_vt_tr)     
    