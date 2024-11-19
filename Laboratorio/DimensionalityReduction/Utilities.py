import seaborn            as sns
import matplotlib.pyplot  as plt


# 1. PRINT THE HEATMAP OF A CORRELATION O COVARIANCE MATRIX
def heatmap_generation(df, cols_name):
    plt.figure(figsize=(18, 18))
    ax = sns.heatmap(
       df, 
       annot    = True, 
       cmap     = 'coolwarm', 
       fmt      = '.2f',
       annot_kws= {'size'  : 9 }, 
       cbar_kws = {'shrink': 0.8}
    )
    ax.set_xticklabels(cols_name, rotation=90)
    ax.set_yticklabels(cols_name, rotation=0)
    plt.show()
    
    
# 2. PRINT SCATTER PLOT
def scatter_generation(title, labels, targets, df, columns, colors):
    plt.figure(figsize=(8, 8))
    plt.xlabel('principal component 1',fontsize=11)
    plt.ylabel('principal component 2',fontsize=11)
    plt.title(title, fontsize=14)
    for i, label in enumerate(labels):
        rows_idx = targets == i
        plt.scatter(df.loc[rows_idx, columns[0]], df.loc[rows_idx, columns[1]], c=colors[i], s=50, label=label)
    plt.legend()
    plt.show()