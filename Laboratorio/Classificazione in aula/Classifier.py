from sklearn.svm             import SVC
from sklearn.neighbors       import KNeighborsClassifier   as KNC
from sklearn.tree            import DecisionTreeClassifier as DTC
from sklearn.ensemble        import RandomForestClassifier as RFC
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes     import CategoricalNB as CNB
import matplotlib.pyplot as     plt
from   seaborn           import heatmap
from sklearn.metrics     import confusion_matrix

classificatori = {
    "svc": SVC,
    "knc": KNC,
    "dtc": DTC,
    "rfc": RFC,
    "cnb": CNB
}

classificatori_nomi = {
    "svc": "SVC",
    "knc": "KNeighborsClassifier",
    "dtc": "DecisionTreeClassifier",
    "rfc": "RandomForestClassifier",
    "cnb": "CategoricalNB"
}


class Classifier:
    def __init__(self, classif_code):
        if classif_code not in classificatori:
            print("Il classificatore non è presente in lista")
            exit(1)
        self.classif_code = classif_code
        self.classif_name = classificatori_nomi[classif_code]
        self.best_score   = None
        self.best_params  = None
        self.final_model  = None
        
    def best_parameters_computing(self, param_grid, scoring, data_obj, cv=5):
        model_template = classificatori[self.classif_code]()
        clf = GridSearchCV(model_template, param_grid, scoring=scoring, cv=cv)
        clf.fit(data_obj.ftx_mtx_tr, data_obj.targ_vt_tr)
        self.best_score  = clf.best_score_
        self.best_params = clf.best_params_
        self.final_model = classificatori[self.classif_code](**self.best_params)
        self.final_model.fit(data_obj.ftx_mtx_tr, data_obj.targ_vt_tr)
        
    def confusion_matrix_computing(self, train_pred, test_pred, data_obj):
        cmtr = confusion_matrix(train_pred, data_obj.targ_vt_tr)
        cmts = confusion_matrix(test_pred,  data_obj.targ_vt_ts)
        fig, axs = plt.subplots(1,2, figsize=(10,5))
        heatmap(
            cmtr, annot=True, xticklabels=data_obj.targ_names, 
            yticklabels=data_obj.targ_names, cmap="Blues", ax=axs[0]
        )
        heatmap(
            cmts, annot=True, xticklabels=data_obj.targ_names, 
            yticklabels=data_obj.targ_names, cmap="Blues", ax=axs[1]
        )
        plt.show()
        
        
        