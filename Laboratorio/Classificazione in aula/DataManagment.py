from sklearn.model_selection import train_test_split

class DataManagment:
    def __init__(self, data_loader, test_size=0.25, tname="Target"):
        data_obj = data_loader()
        # matrice delle features
        self.ftx_mtx = data_obj.data
        # vettore delle classi
        self.target_vt  = data_obj.target
        self.ftx_names  = data_obj.feature_names
        self.targ_names = data_obj.target_names if "target_names" in data_obj else tname
        self.ftx_mtx_tr = None
        self.ftx_mtx_ts = None
        self.targ_vt_tr = None
        self.targ_vt_ts = None
        self.train_test_conf(test_size)
        
    def train_test_conf(self, test_size):
        self.ftx_mtx_tr, self.ftx_mtx_ts, self.targ_vt_tr, self.targ_vt_ts = \
        train_test_split(
            self.ftx_mtx, 
            self.target_vt, 
            test_size=test_size, 
            random_state=42
        )