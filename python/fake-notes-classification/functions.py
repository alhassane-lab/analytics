# Library of Functions for the OpenClassrooms Multivariate Exploratory Data Analysis Course

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from pandas.plotting import parallel_coordinates
import seaborn as sns


palette = sns.color_palette("bright", 10)

def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    """Display correlation circles, one for each factorial plane"""

    # For each factorial plane
    for d1, d2 in axis_ranks: 
        if d2 < n_comp:

            # Initialise the matplotlib figure
            #fig, ax = plt.subplots(figsize=(8,8))

            # Determine the limits of the chart
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # Add arrows
            # If there are more than 30 arrows, we do not display the triangle at the end
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:], 
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (see the doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
            
            # Display variable names
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)
            
            # Display circle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # Define the limits of the chart
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        
            # Display grid lines
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # Label the axes, with the percentage of variance explained
            plt.xlabel('PC{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('PC{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Correlation Circle (PC{} and PC{})".format(d1+1, d2+1))
            #plt.show(block=False)
        
def display_factorial_planes(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None):
    '''Display a scatter plot on a factorial plane, one for each factorial plane'''

    # For each factorial plane
    for d1,d2 in axis_ranks:
        if d2 < n_comp:
 
            # Initialise the matplotlib figure      
            #fig = plt.figure(figsize=(7,6))
        
            # Display the points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value)
                plt.legend()

            # Display the labels on the points
            if labels is not None:
                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):
                    plt.text(x, y, labels[i],
                              fontsize='14', ha='center',va='center') 
                
            # Define the limits of the chart
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1
            plt.xlim([-boundary,boundary])
            plt.ylim([-boundary,boundary])
        
            # Display grid lines
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # Label the axes, with the percentage of variance explained
            plt.xlabel('PC{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('PC{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Projection of points (on PC{} and PC{})".format(d1+1, d2+1))
            #plt.show(block=False)
   
def display_scree_plot(pca):
    '''Display a scree plot for the pca'''

    scree = pca.explained_variance_ratio_*100
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("Number of principal components")
    plt.ylabel("Percentage explained variance")
    plt.title("Scree plot")
    plt.show(block=False)

def append_class(df, class_name, feature, thresholds, names):
    '''Append a new class feature named 'class_name' based on a threshold split of 'feature'.  Threshold values are in 'thresholds' and class names are in 'names'.'''
    
    n = pd.cut(df[feature], bins = thresholds, labels=names)
    df[class_name] = n

def plot_dendrogram(Z, names):
    '''Plot a dendrogram to illustrate hierarchical clustering'''

    #plt.figure(figsize=figsize)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('distance')
    dendrogram(
        Z,
        labels = names,
        orientation = "left",
    )
    #plt.show()

def addAlpha(colour, alpha):
    '''Add an alpha to the RGB colour'''
    
    return (colour[0],colour[1],colour[2],alpha)

def display_parallel_coordinates(df, num_clusters):
    '''Display a parallel coordinates plot for the clusters in df'''

    # Select data points for individual clusters
    cluster_points = []
    for i in range(num_clusters):
        cluster_points.append(df[df.cluster==i])
    
    # Create the plot
    fig = plt.figure(figsize=(12, 15))
    title = fig.suptitle("Parallel Coordinates Plot for the Clusters", fontsize=18)
    fig.subplots_adjust(top=0.95, wspace=0)

    # Display one plot for each cluster, with the lines for the main cluster appearing over the lines for the other clusters
    for i in range(num_clusters):    
        plt.subplot(num_clusters, 1, i+1)
        for j,c in enumerate(cluster_points): 
            if i!= j:
                pc = parallel_coordinates(c, 'cluster', color=[addAlpha(palette[j],0.2)])
        pc = parallel_coordinates(cluster_points[i], 'cluster', color=[addAlpha(palette[i],0.5)])

        # Stagger the axes
        ax=plt.gca()
        for tick in ax.xaxis.get_major_ticks()[1::2]:
            tick.set_pad(20)        


def display_parallel_coordinates_centroids(df, num_clusters):
    '''Display a parallel coordinates plot for the centroids in df'''

    # Create the plot
    fig = plt.figure(figsize=(12, 5))
    title = fig.suptitle("Parallel Coordinates plot for the Centroids", fontsize=18)
    fig.subplots_adjust(top=0.9, wspace=0)

    # Draw the chart
    parallel_coordinates(df, 'cluster', color=palette)

    # Stagger the axes
    ax=plt.gca()
    for tick in ax.xaxis.get_major_ticks()[1::2]:
        tick.set_pad(20)
        
# On définit une fonction qui check les doublons dans une colonne et nous indique le nombre de doublons contenus dans cette colonne suivi d'un aperçu de ces doulons.
def check_duplicates(data, column):
    n = len(data.index)
    if len(data[column].drop_duplicates()) == n :
        print ("La colonne", column, "ne contient pas de doublons")
    else :
        if len(column) == 1:
            print("La clé", column, "contient ", len(data[data[column].duplicated()]), " doublons.")
        else:
            print("La clé composée des colonnes", column, "contient ", len(data[data[column].duplicated()]), " doublons.")
        return data[data[column].duplicated()]

# On définit une fonction qui check les valeurs manquantes et indique le nombre de valeurs manquantes contenues dans cette colonne suivi d'un aperçu de ces valeurs manquantes.
def check_nan(data, column):
    print(data.isnull().sum())
    if len(data[data[column].isnull()]) == 0:
        print ("La colonne", column, "ne contient aucune valeur manquante")
    else :
        print("La colonne", column, "contient ", len(data[data[column].isnull()]), " valeurs manquantes.")
        return data[data[column].isnull()]


# On définit une fonction qui detecter les outliers
outliers=[]
def check_outliers(data):
    threshold = 2
    mean = np.mean(data)
    std = np.std(data)
    
    for i in data:
        z_score = (i- mean)/std
        if z_score > threshold:
            outliers.append(i)
    print("Le nombre d'outliers détectés  " + str(len(outliers)))


def df_abstract(df):
    print('='*70)
    print('Apérçu des données :')
    display(df.head())
    print('(Lignes, Colonnes) : \t', df.shape)
    print()
    print('='*70)
    print('\n Infos : \n')
    df.info()
    print()
    print('='*70)
    print('\n Describe :')
    display(df.describe())
    print()
    print('='*70)
    
# On définit une fonction qui check les doublons dans une colonne et nous indique le nombre de doublons contenus dans cette colonne suivi d'un aperçu de ces doulons.
def check_duplicates(data, column):
    n = len(data.index)
    if len(data[column].drop_duplicates()) == n :
        print ("La colonne", column, "ne contient pas de doublons")
    else :
        if len(column) == 1:
            print("La clé", column, "contient ", len(data[data[column].duplicated()]), " doublons.")
        else:
            print("La clé composée des colonnes", column, "contient ", len(data[data[column].duplicated()]), " doublons.")
        return data[data[column].duplicated()]

# On définit une fonction qui check les valeurs manquantes et indique le nombre de valeurs manquantes contenues dans cette colonne suivi d'un aperçu de ces valeurs manquantes.
def check_nan(data, column):
    if len(data[data[column].isnull()]) == 0:
        print ("La colonne", column, "ne contient aucune valeur manquante")
    else :
        print("La colonne", column, "contient ", len(data[data[column].isnull()]), " valeurs manquantes.")


# On définit une fonction qui detecter les outliers




def check_outliers(data, threshold = 2):
    outliers=[]
    mean = np.mean(data)
    std = np.std(data)
    
    for i in data:
        z_score = (i- mean)/std
        if z_score > threshold:
            outliers.append(i)
    print("Le nombre d'outliers détectés  " + str(len(outliers)))
    return outliers



def cles_potentielles(df, max_allowed=10):
    from itertools import chain, combinations
    combi_list = chain.from_iterable( combinations(list(df), x) for x in range(1, len(list(df))+1) )
    found = 0
    for candidate in combi_list:
        tmp = df.drop_duplicates(candidate)
        if len(tmp) == len(df):
            print( list(candidate) )
            found +=1
        if found > max_allowed:
            print( 'Nombre maximum autorisé atteint.', end=' ')
            print( 'Veuillez augmenter cette valeur si vous voulez rechercher davantage de clés primaires candidates.' )
            return
    if found == 0:
        print('''Aucune clé primaire, simple ou composée, n'a pu être trouvée ! Il y a forcément des doublons.''')
        

from scipy.stats import chi2_contingency
import numpy as np
import seaborn as sns
from scipy.stats import shapiro
import scipy.stats as st
from matplotlib import pyplot as plt
import researchpy


def shapiro_test(x, alpha=0.05):
    x1, pval1 = st.shapiro(x)
    print("=" * 100, "\n")
    print("\t\t\t\t\t TEST DE LA NORMALITE (TEST DE SHAPIRO) \n")
    print("=" * 100, "\n")
    print("""
    \t##### \033[1m0. Hypothèse du test\033[0m #####\n
    H0 : \033[1m{0}\033[0m suit une loi normale \n
    H1 : \033[1m{0}\033[0m ne suit pas une loi normale \n
    \t##### \033[1m1. Paramètre du test de Shapiro\033[0m #####\n
    Variable aléatoire étudiée : \033[1m{0}\033[0m\n
    Indice de confiance : \033[1m{1}\033[0m\n
    Taille de l'échantillon : \033[1m{2}\033[0m\n
    \t #### \033[1m2. Résultat du test\033[0m ####\n
    p-value de shapiro : \033[1m{3}\033[0m\n
    coefficient de shapiro : \033[1m{4}\033[0m\n 
    \t #### \033[1m3. Conclusion du test\033[0m ####\n""".format(x.name, alpha, x.shape[0], pval1, x1))
    if pval1 < alpha:
        print("L'hypothèse nulle est rejetée \t ==> \033[1m{}\033[0m ne suit pas une loi normale".format(x.name))
    else:
        print("On ne peut pas rejeter l'hypothèse nulle H0 (\033[1m{}\033[0m suit une loi normale)".format(x.name))
    print()
    print("=" * 100, "\n")


def spearman_test(x, y, alpha=0.05):
    pvalue = st.spearmanr(x, y)[1]
    rs = st.spearmanr(x, y)[0]
    print("=" * 100, "\n")
    print("\t\t\t\t\t  \033[1mTEST D'INDEPENDANCE DE SPEARMAN \033[0m \n")
    print("=" * 100, "\n")
    print("""\t##### \033[1m0. Hypothèse du test\033[0m #####\n
    H0 : Les variables {0} sont indépendantes\n
    H1 : Les variables {0} sont corrélées\n
    \t##### \033[1m1. Paramètre du test de Shapiro\033[0m #####\n
    Variables aléatoires étudiées : \033[1m{0}\033[0m\n
    Indice de confiance : \033[1m{1}\033[0m\n
    Taille de l'échantillon : \033[1m{2}\033[0m\n
    \t #### \033[1m2. Résultat du test\033[0m ####\n
    coefficient de Spearman : \033[1m{3}\033[0m\n
    p-value associée au test de Spearman : \033[1m{4}\033[0m\n
    \t #### \033[1m3. Conclusion du test\033[0m ####\n""".format((x.name, y.name), alpha, x.shape[0], rs, pvalue))

    if abs(rs) < .10:
        qual = 'négligeable (ou nulle)'
    elif abs(rs) < .20:
        qual = 'faible'
    elif abs(rs) < .40:
        qual = 'modérée'
    elif abs(rs) < .60:
        qual = 'plutôt forte'
    elif abs(rs) < .80:
        qual = 'forte'
    else:
        qual = 'très forte'
    print()
    if rs == 0:
        print(" --> On ne peut pas rejeter l'hypothèse nulle H0 (Les variables sont indépendantes)")
    elif rs < 0:
        if pvalue < alpha:
            print(
                """ --> \033[1m{}\033[0m présentent \033[1msignificativement\033[0m une \033[1m{}\033[0m corrélation négative.""".format(
                    (x.name, y.name), qual))
        else:
            print(" --> \033[1m{}\033[0m présentent une corrélation négative \033[1m{} peu significative\033[0m".format(
                (x.name, y.name), qual))
    elif rs > 0:
        if pvalue < alpha:
            print(
                " --> \033[1m{}\033[0m présentent \033[1msignificativement\033[0m une \033[1m{}\033[0m corrélation positive.".format(
                    (x.name, y.name), qual))
        else:
            print(" --> \033[1m{}\033[0m présentent une corrélation {} positive \033[1mpeu significative\033[0m".format(
                (x.name, y.name), qual))


def chi2test(data, x, y, alpha=0.05):
    # Tableau de contingence
    cont = data[[x, y]].pivot_table(index=x, columns=y, aggfunc=len, margins=False)

    # Test du chi-2
    chi2, p, dof, expected = chi2_contingency(cont, alpha)

    print("=" * 100, "\n")
    print("""\t\t\t\t\t \033[1mTEST D'INDEPENDANCE DU CHI-2\033[0m \n""")
    print("=" * 100, "\n")

    f, ax = plt.subplots(1, 2, figsize=(15, 6))
    sns.heatmap(cont, annot=True, fmt="d", linewidths=2, linecolor="k", ax=ax[0])
    sns.heatmap(np.int64(expected), annot=True, fmt="d", linewidths=2, ax=ax[1], linecolor="k")
    ax[0].set_title("Fréquence observées")
    ax[1].set_title("Fréquence théoriques")
    plt.show()
    print("""
    \t\t ##### \033[1m0. Hypothèse du test \033[0m ##### \n
    H0 : \033[1m{0}\033[0m et \033[1m{1}\033[0m sont indépendantes \n
    H1 : \033[1m{0}\033[0m et \033[1m{1}\033[0m sont corrélées \n 
    \n\t\t ##### \033[1m1. Paramètre du test\033[0m ##### 
    Variables aléatoires étudiées : \033[1m{0} et {1}\033[0m\n
    Indice de confiance alpha : \033[1m{2}\033[0m \n
    Degré de liberté : \033[1m{3}\033[0m\n
    \n\t\t ##### \033[1m2. Résultat du test du Qui-2 \033[0m ##### \n
    Coefficient du qui-2 : \033[1m{4}\033[0m\n
    p-value calculée : \033[1m{5}\033[0m\n
    \n\t\t ##### \033[1m3. Conclusion du test \033[0m ##### 
    """.format(x, y, alpha, dof, chi2, p))

    if p < alpha:
        print("H0 est rejetée : \033[1m{}\033[0m et \033[1m{}\033[0m sont corrélées significativement".format(x, y))
    else:
        print("H1 est rejetée : \033[1m{}\033[0m et \033[1m{}\033[0m ne sont pas corrélées entre elles".format(x, y))

    # Test de V Cramer
    print("\n\n")
    print("=" * 100, "\n")
    print("""\t\t\t\t\t \033[1mTEST DE SIGNIFICATIVITE DE V CRAMER\033[0m \n""")
    print("=" * 100, "\n")

    crosstab, res = researchpy.crosstab(data[x], data[y], test='chi-square')
    coeff_cramer = res.iloc[2, 1]
    if abs(coeff_cramer) < .10:
        qual = "L'intensité du lien entre les variables est \033[1mquasiement nulle\033[0m"
    elif abs(coeff_cramer) < .20:
        qual = "L'intensité du lien entre les variables est \033[1mfaible\033[0m"
    elif abs(coeff_cramer) < .30:
        qual = "L'intensité du lien entre les variables est \033[1mmoyen\033[0m"
    else:
        qual = "L'intensité du lien entre les variables est \033[1mforte\033[0m"
    print("""Le coefficient de Cramer est de : \033[1m{}\033[0m \n
    {}""".format(coeff_cramer, qual))
    print("\n")
    print("=" * 100, "\n")
    
from IPython.display import HTML
import random

def hide_toggle(for_next=False):
    this_cell = """$('div.cell.code_cell.rendered.selected')"""
    next_cell = this_cell + '.next()'

    toggle_text = 'Toggle show/hide'  # text shown on toggle link
    target_cell = this_cell  # target cell to control with toggle
    js_hide_current = ''  # bit of JS to permanently hide code in current cell (only when toggling next cell)

    if for_next:
        target_cell = next_cell
        toggle_text += ' next cell'
        js_hide_current = this_cell + '.find("div.input").hide();'

    js_f_name = 'code_toggle_{}'.format(str(random.randint(1,2**64)))

    html = """
        <script>
            function {f_name}() {{
                {cell_selector}.find('div.input').toggle();
            }}

            {js_hide_current}
        </script>

        <a href="javascript:{f_name}()">{toggle_text}</a>
    """.format(
        f_name=js_f_name,
        cell_selector=target_cell,
        js_hide_current=js_hide_current, 
        toggle_text=toggle_text
    )

    return HTML(html)
