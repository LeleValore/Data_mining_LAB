# Classe utilizzata per il calcolo della correlazione. 
# Esistono diverse tecniche di correlazione
#  - Pearson: 
#    Tipo di correlazione: Lineare
#    La correlazione di Pearson valuta la relazione lineare tra due variabili continue. E' sensibile agli outlier e assume che le #    variabili siano distribuite normalmente e che la relazione tra di esse sia lineare.
# 
#  - Sperman:
#    Tipo di correlazione: Monotona
#    La correlazione di Spearman valuta la relazione monotona tra due variabili ordinate. Invece di considerare i valori effettivi #    delle variabili, considera il loro ordine (rango) nei dati.
#    La correlazione di Spearman è robusta agli outlier ed è utile quando le relazioni tra variabili non sono necessariamente 
#    lineari.
#
#  - Kendall:
#    Tipo di correlazione: Monotona
#    La correlazione di Kendall valuta la relazione monotona tra due variabili ordinate. Come la correlazione di Spearman, la 
#    correlazione di Kendall è robusta agli outlier ed è particolarmente utile quando si tratta di dati ordinali (dati che possono 
#    essere classificati in un ordine specifico) o quando non si vogliono fare ipotesi sulla distribuzione dei dati.


from Utilities import heatmap_generation


correlations_type = {
     "pr": "pearson",
     "sp": "spearman",
     "kd": "kendall"
}

class Correlator:
    def __init__(self, df, corr_type):
        self.columns     = df.columns
        self.cor_type    = correlations_type[corr_type] + " correlation"
        self.correlation = df.corr(method=correlations_type[corr_type])
        
    def correlation_heatmap(self):
        heatmap_generation(self.correlation, self.columns)