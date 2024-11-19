# Covarianza:
# - misura la relazione lineare tra due variabili, ossia la tendenza delle due variabili a variare insieme.
# - covarianza positiva indica che le due variabili tendono a crescere insieme, negativa indica che una variabile tende a 
#   diminuire quando l'altra aumenta; o vicina a zero (indica una debole relazione lineare).
#
# In skleanr:
# - EmpiricalCovariance: calcola la covarianza secondo la formula e l'approccio standard. usata quando si ha un numero sufficiente
#   di campioni, inoltre non si sa nulla sul rumore e gli outlier.
# - Ledoit-Wolf Estimator: usata quando si ha un piccolo set di campioni, o quando i dati sono molto rumorosi.
# - Minimum Covariance Determinant (MinCovDet): dataset contine outlier.


# IMPORT SECTION
from   itertools          import combinations
from   sklearn.covariance import EmpiricalCovariance
from   Utilities          import heatmap_generation


# CLASS
class Covarianzer:
    def __init__(self, df):
        self.cols           = list(df.columns)
        self.covariance_mtx = self.covariance_computing(df)
    
    def covariance_computing(self, df):
        cov = EmpiricalCovariance(assume_centered=True).fit(df.values)
        return cov
    
    def covariance_heatmap(self):
        heatmap_generation(
            self.covariance_mtx.covariance_,
            self.cols
        )        