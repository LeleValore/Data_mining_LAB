import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler


# DIZIONARIO CONTENENTE LE CLASSI DI NORMALIZZAZIONE IN PYTHON
normalizetors = {
    "ss": ("StandardScaler", StandardScaler),
    "mm": ("MinMaxScaler"  , MinMaxScaler  ),
    "ma": ("MaxAbsScaler"  , MaxAbsScaler  ),
    "rs": ("RobustScaler"  , RobustScaler  ),
}


# DEFINIZIONE DELLA CLASSE
class Normalization:
    def __init__(self, model, df, quantile_range=None):
        norm_tuple = normalizetors[model]
        self.normalization_name = norm_tuple[0] 
        self.normalizator_obj   = self.set_normalization_obj(quantile_range, norm_tuple)
        self.data  = self.data_normalization(df)      
       
    def set_normalization_obj(self, quantile_range, norm_tuple):
        if not quantile_range:
            return norm_tuple[1]() # se non è specificato il quantile_range allora costruttore di default senza parametro
        return norm_tuple[1](quantile_range=quantile_range) # senno setta il quantile range a quanto è stato richiesto "Caso del rs"
    
    def data_normalization(self, df):
        data_normalizate = pd.DataFrame(
            self.normalizator_obj.fit_transform(df.values) # dati passati come array numpy vengono convertiti in dataframe di nuovo
        )
        data_normalizate.columns = df.columns # riassegno colonne 
        return data_normalizate # ritorno il dataframe normalizzato 