# Gerarchici
from sklearn.cluster         import AgglomerativeClustering

# Densità
from sklearn.cluster         import DBSCAN, OPTICS, HDBSCAN

# Partizionali
from sklearn_extra.cluster   import KMedoids
from sklearn.cluster         import KMeans

import itertools
from   sklearn.metrics       import silhouette_score
import pandas as pd
from   sklearn.decomposition import PCA
import matplotlib.pyplot     as plt
import json


cluters_type = {
    "km": KMeans,
    "ka": KMedoids,
    "gc": AgglomerativeClustering,
    "db": DBSCAN,
    "op": OPTICS,
    "hdb":HDBSCAN
}

clusters_name = {
    "km": "KMeans",
    "ka": "KMedoids",
    "gc": "AgglomerativeClustering",
    "db": "DBSCAN",
    "op": "OPTICS",
    "hdb":"HDBSCAN"
}


class Clusterizer:
    def __init__(self, clt_key="km"):
        if clt_key not in clusters_name:
            print("La chiave non è presente")
            return
        self.name    = clusters_name[clt_key]
        self.clt_key = clt_key
        self.models  = []
        self.selected_models=[]
        self.qom = None
        
    def models_configuration(self, matrix, params=None):
        # HOMEWORK: GESTIRE IL CASO params=None PER EVITARE
        #           ERRORI. QUINDI METTERE UN IF-ELSE
        # QUESTA PARTE COSTITUISCE L'ELSE
        params_keys   = list(params.keys())
        params_values = params.values()
        # CREA TUTTE LE POSSIBILI COMBINAZIONI DEI PARAMETRI
        # PASSATI.
        models_comb   = itertools.product(*params_values)
        models_params = []
        for model_comb in models_comb:
            model_comb_dict = {}
            for idx, key_name in enumerate(params_keys):
                model_comb_dict[key_name]=model_comb[idx] 
            models_params.append(model_comb_dict)
        for comb_dict in models_params:
            self.models.append((
                comb_dict,
                cluters_type[self.clt_key](**comb_dict).fit(matrix)
            ))
            
    def silouette_evaluation(self, matrix):
        self.qom = []
        for conf_par, model in self.models:
            try:
                s_score = silhouette_score(matrix, model.labels_)
            except:
                s_score = 0
            conf_par["s_score"] = s_score
            self.qom.append(conf_par)
        self.qom = pd.DataFrame(self.qom)
        self.qom.sort_values(by="s_score", ascending = False, inplace=True)
        condizione = self.qom["s_score"] > 0.5
        self.qom = self.qom[condizione]
        
    def models_selection(self, models_idx):
        for idx in models_idx:
            self.selected_models.append(self.models[idx])
   

    def models_printing(self, matrix):
        pca = PCA(n_components=2)
        matrix_red = pca.fit_transform(matrix)
        for params, model in self.selected_models:
            fig = plt.figure(figsize=(5,5))
            plt.scatter(matrix_red[:,0], matrix_red[:,1], c=model.labels_)
            plt.title(json.dumps(params) + "\n")
            plt.show()
            print("\n\n")
        
        
            