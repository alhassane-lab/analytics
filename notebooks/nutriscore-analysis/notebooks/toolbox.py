import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
import warnings
import missingno as msno
from wordcloud import WordCloud
from scipy.stats import chi2_contingency
from scipy.stats import shapiro
import scipy.stats as st
import researchpy
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram
from pandas.plotting import parallel_coordinates
from matplotlib.collections import LineCollection



def null_factor(df, rate_threshold=80):
    nan_rate = ((df.isnull().sum() / df.shape[0])*100).sort_values(ascending=False).reset_index()
    nan_rate.columns = ['variable','nan_rate']
    nan_rate2 = nan_rate[nan_rate['nan_rate'] >= rate_threshold]
    print('Nombre de colonnes 100% nulles :', (len(nan_rate2)))  
    return nan_rate2


def top_words(df, column="countries_en"):
    count_keyword = dict()
    for index, col in df[column].items():
        if isinstance(col, float):
            continue
        for word in col.split(','):
            if word in count_keyword.keys():
                count_keyword[word] += 1
            else :
                count_keyword[word] = 1
    
    keyword_top = []
    for k,v in count_keyword.items():
        keyword_top.append([k,v])
    keyword_top.sort(key = lambda x:x[1], reverse = True)
    df = pd.DataFrame(keyword_top,columns=[column,"occurrence"])
    return df , keyword_top


def plot_world_cloud(df,column="categories_en", top=100,figsize=(12,6)):
    fig = plt.figure(1, figsize)
    ax1 = fig.add_subplot(1,1,1)
    words = dict()
    ok, trunc_occurences = top_words(df=df, column=column)
    for s in trunc_occurences[:top]:
        words[s[0]] = s[1]

    word_cloud = WordCloud(width=900,height=500, normalize_plurals=False,
                        background_color="white")
    word_cloud.generate_from_frequencies(words)
    ax1.imshow(word_cloud, interpolation="bilinear")
    ax1.axis('off')
    plt.title("Word cloud top {} occurrences {}\n".format(top,column))
    plt.show()

def split_words(df, column = 'countries_en'):
    list_words = set()
    for word in df[column].str.split(','):
        if isinstance(word, float):
            continue
        list_words = set().union(word, list_words)
    return list_words


#this functiun aims at filtring data according to a suffix in column name 
def search_componant(df, suffix):
    componants = [x for x in df.columns if suffix in x]
    return componants



def redundant_col(df):
    redundant_columns = []
    for col in df.columns:
        if "_en" in col:
            no_suffix = col.replace('_en','')
            tags = col.replace('_en','_tags')
            en = col.replace('_tags','_en')
            print("{:<20}|  no suffixe  : {}    |   suffixe _en : {}    |  suffixe _tags : {}".format(no_suffix,
                                                                        no_suffix in df.columns, en in df.columns,tags in df.columns))
            if en in df.columns and no_suffix in df.columns: 
                redundant_columns.append(no_suffix)
            if tags in df.columns and no_suffix or en in df.columns: 
                redundant_columns.append(tags)
                
    return redundant_columns


# this aims at returning a dataframe containing columns with a filling rate equal or greater than a specific threshold
def null_factor(df, rate_threshold=80):
    nan_rate = ((df.isnull().sum() / df.shape[0])*100).sort_values(ascending=False).reset_index()
    nan_rate.columns = ['variable','nan_rate']
    nan_rate2 = nan_rate[nan_rate['nan_rate'] >= rate_threshold]
    return nan_rate2


# this function aims at vizualising the filling rate, it requires null factor function
def filling_rate_viz(df, del_threshold=20):
    filling_features = null_factor(df, 0)
    filling_features["nan_rate"] = 100-filling_features["nan_rate"]
    filling_features = filling_features.sort_values("nan_rate", ascending=False) 

    fig = plt.figure(figsize=(20, 35))

    sns.barplot(x="nan_rate", y="variable", data=filling_features, palette='mako')
    #Seuil pour suppression des varaibles
    plt.axvline(x=del_threshold, linewidth=2, color = 'r')
    plt.text(del_threshold, 65, 'deletion threshold', fontsize = 15, color = 'r')

    plt.title("Data filling rate")
    plt.xlabel("filling rate (%)")
    plt.show()
   


# Cette fonction a pour objectif d'ouvrir un fichier en onction de son extension et de le return sous forme de dataframe
def read_data(file_extension, path):
    if file_extension == 'xlsx':
        data = pd.read_excel(path, engine='openpyxl')
    elif file_extension == 'xls':
        data = pd.read_excel(path)
    elif file_extension == 'csv':
        data = pd.read_csv(path)           
    return data


# Cette fonction a pour objectif d'afficher un aperçu et une description d'un dataframe ainsi que le nbre de missing values qu'il contient
def describe_data(df, figsize=(6,4)):
    print('*'*35,'Data infos','*'*35)
    #df.info()
    #print()
    
    #Check nombre de colonnes
    print("Nombre de colonnes : ",df.shape[1],"\n")

    #Check nombre de lignes
    print("Nombre de lignes : ",df.shape[0],"\n")
    
    # Analyse des valeurs manquantes
    plt.figure(figsize=(12,8))
    print('\n','*'*34,"Valeurs manquantes",'*'*34)
    all_df = df.isnull().sum().sum(), df.notnull().sum().sum()
    plt.pie(all_df, autopct='%1.1f%%', shadow=False, startangle=90,labels=['Missing values', 'Not missing values'], explode = (0, 0.02), colors=["lightblue","steelblue"], pctdistance=0.4, labeldistance=1.1)
    circle = plt.Circle( (0,0), 0.65, color='white')
    p=plt.gcf()
    p.gca().add_artist(circle)
    plt.show()
    
    print("Nombre total de valeurs manquantes : ",df.isna().sum().sum(),'\n')


    
# Cette fonction a pour objectif de visualiser et d'afficher les statistques d'un indicateur donnée par rapport à un bloc géographique
def univariate(df, var_list, region_col, year_col, region, year):
    palette =["steelblue","lightblue",]
    df = df[(df[region_col]==region) & (df['year_col']==year)]
    print("*"*25,'\033[1m',region, year,'\033[0m',"*"*25,"\n")
    for var in var_list:
        print("Indicateur",'\033[1m' , var,'\033[0m',"\n")
        f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw= {"height_ratios": (0.2, 1)},)
        mean=df[var].mean()
        median=df[var].median()
        mode=df[var].mode().values[0]

        sns.boxplot(data=df, x=var, ax=ax_box)
        ax_box.axvline(mean, color='r', linestyle='--')
        ax_box.axvline(median, color='g', linestyle='-')
        ax_box.axvline(mode, color='b', linestyle='-')

        sns.histplot(data=df, x=var, ax=ax_hist, kde=True, bins=50)
        ax_hist.axvline(mean, color='r', linestyle='--', label="Mean")
        ax_hist.axvline(median, color='g', linestyle='-', label="Median")
        ax_hist.axvline(mode, color='b', linestyle='-', label="Mode")

        ax_hist.legend()

        ax_box.set(xlabel='')
        plt.show()

        print ('\033[1m',"Moyenne :",'\033[0m', round(df[var].mean(), 2))
        print ('\033[1m',"Médiane :",'\033[0m', round(df[var].median(), 2))
        print ('\033[1m',"Écart-type :",'\033[0m', round(df[var].kurtosis(), 2))
        print("\n","-"*10,"\n")
        
        
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
        
        
def infos_columns(df):
    print('*'*26,"Nombre de valeurs uniques par colonne", '*'*26,'\n')
    for column in list(df):
        print(column, " : ",len(df[column].unique()),'\n')
        
              
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


def stats(data, region_col, indicators):
    # On parcourt les regions
    for region in data[region_col].unique():
        # On initialise un dict avec la colonne qui indique les indicateurs statistiques à calculer
        stats = {'Indicateur statistique':['mean','median','std','mode', 'kurtosis']}
        # On parcourt les indicateurs pertinents
        for indicator in indicators:
            # On calcule les stats 
            mean = data[data[region_col]==region][indicator].mean()
            median = data[data[region_col]==region][indicator].median()
            mode = data[data[region_col]==region][indicator].mode()[0]
            std = data[data[region_col]==region][indicator].std()
            kurtosis = data[data[region_col]==region][indicator].kurt()
            # On met à jour le dictionnaire avec les 
            stats.update({indicator:[mean,median,std,mode,kurtosis]})
        stats2=pd.DataFrame(stats)
        print("\n","*"*75,region,"*"*75)
        display(stats2.round(2))
        

        
def stats_viz(df, indicators, region_col, region):
    palette =["steelblue","lightblue",]
    print("*"*25,'\033[1m',region,'\033[0m',"*"*25)
    for var in indicators:
        print("\n","Indicateur",'\033[1m' , var,'\033[0m',"\n")
        f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw= {"height_ratios": (0.2, 1)},)
        mean=df[var].mean()
        median=df[var].median()
      
        sns.boxplot(data=df, x=var, ax=ax_box, showfliers = True)
        ax_box.axvline(mean, color='r', linestyle='--')
        ax_box.axvline(median, color='g', linestyle='-')
        
        sns.histplot(data=df, x=var, bins=50, ax=ax_hist,)
        ax_hist.axvline(mean, color='r', linestyle='--', label="Mean")
        ax_hist.axvline(median, color='g', linestyle='-', label="Median")
       
        #ax_hist.legend()

       # ax_box.set(xlabel='')
        plt.show()

        print ('\033[1m',"Moyenne :",'\033[0m', round(df[var].mean(), 2))
        print ('\033[1m',"Médiane :",'\033[0m', round(df[var].median(), 2))
        print ('\033[1m',"Écart-type :",'\033[0m', round(df[var].std(), 2))
        

        
def stats_region_year(df, indicators, region_col, year_col, region, year):
    palette =["steelblue","lightblue",]
    df = df[(df[region_col]==region) & (df[year_col]==year)]
    print("*"*25,'\033[1m',region,'\033[0m',"*"*25)
    
    for var in indicators:
        print("\n","Indicateur",'\033[1m' , var,'\033[0m',"\n")
        f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw= {"height_ratios": (0.2, 1)},)
        mean=df[var].mean()
        median=df[var].median()
      
        sns.boxplot(data=df, x=var, ax=ax_box, showfliers = True)
        ax_box.axvline(mean, color='r', linestyle='--')
        ax_box.axvline(median, color='g', linestyle='-')
        
        sns.histplot(data=df, x=var, bins=50, ax=ax_hist,)
        ax_hist.axvline(mean, color='r', linestyle='--', label="Mean")
        ax_hist.axvline(median, color='g', linestyle='-', label="Median")
       
        #ax_hist.legend()

       # ax_box.set(xlabel='')
        plt.show()

        print ('\033[1m',"Moyenne :",'\033[0m', round(df[var].mean(), 2))
        print ('\033[1m',"Médiane :",'\033[0m', round(df[var].median(), 2))
        print ('\033[1m',"Écart-type :",'\033[0m', round(df[var].std(), 2))
        
        

def shapiro_test(x, alpha=0.05):
    x1, pval1 = shapiro(x)
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
    \t##### \033[1m1. Paramètre du test\033[0m #####\n
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
    cont = cont.fillna(0)
    
    # Test du chi-2
    chi2, p, dof, expected = chi2_contingency(cont, alpha)

    print("=" * 100, "\n")
    print("""\t\t\t\t\t \033[1mTEST D'INDEPENDANCE DU CHI-2\033[0m \n""")
    print("=" * 100, "\n")

    f, ax = plt.subplots(1, 2, figsize=(15, 6))
    sns.heatmap(cont, annot=True, linewidths=2, linecolor="k", ax=ax[0], cmap=sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True),fmt='g' )
    sns.heatmap(np.int64(expected), annot=True, fmt="d", linewidths=2, ax=ax[1], linecolor="k",cmap=sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True))
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
    


def pearson_test(x, y, alpha=0.05):
    pvalue = st.pearsonr(x, y)[1]
    rs = st.pearsonr(x, y)[0]
    print("=" * 100, "\n")
    print("\t\t\t\t\t  \033[1mTEST D'INDEPENDANCE DE PEARSON \033[0m \n")
    print("=" * 100, "\n")
    print("""\t##### \033[1m0. Hypothèse du test\033[0m #####\n
    H0 : Les variables {0} sont indépendantes\n
    H1 : Les variables {0} sont corrélées\n
    \t##### \033[1m1. Paramètre du test \033[0m #####\n
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