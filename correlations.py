import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# Pour afficher les graphiques directement dans Colab
%matplotlib inline

# Étape 1 : Charger les données depuis le fichier CSV
data = pd.read_csv("DF_EVOL_TENSE_FLORIAN.csv")

# Afficher les premières lignes pour vérifier
print("Aperçu des données :")
print(data.head())

# Étape 2 : Prétraitement des données
# Convertir la colonne 'date' en format datetime
data['date'] = pd.to_datetime(data['date'], errors='coerce')

# Extraire l'année de la colonne 'date' et créer une nouvelle colonne 'Année'
data['Année'] = data['date'].dt.year

# Vérifier les valeurs manquantes dans 'Année'
missing_years = data['Année'].isnull().sum()
if missing_years > 0:
    print(f"Attention : {missing_years} entrées ont une date invalide et seront supprimées.")
    data = data.dropna(subset=['Année'])

# Convertir 'Année' en entier
data['Année'] = data['Année'].astype(int)

# Afficher les années uniques disponibles
print("\nAnnées disponibles dans les données :", data['Année'].unique())

# Définir les colonnes correspondant aux temps verbaux
temps_verbaux = ['Présent', 'Imparfait', 'Passé simple', 'Passé composé', 'Futur', 'Plus que parfait']

# Vérifier que toutes les colonnes existent dans les données
for tense in temps_verbaux:
    if tense not in data.columns:
        raise ValueError(f"La colonne '{tense}' n'est pas présente dans les données.")

# Vérifier la présence de la colonne 'Année' après extraction
if 'Année' not in data.columns:
    raise ValueError("La colonne 'Année' n'a pas été créée correctement.")

# Étape 3 : Calculer et visualiser la matrice de corrélation
correlation_matrix = data[temps_verbaux].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Matrice de corrélation entre les temps verbaux")
plt.show()

# Étape 4 : Calculer l'évolution des corrélations
# Initialiser un DataFrame pour stocker les corrélations
evolution_corr_ps = pd.DataFrame()
evolution_corr_ps['Année'] = sorted(data['Année'].unique())

# Initialiser les colonnes pour les corrélations
for tense in temps_verbaux:
    if tense != 'Passé simple':
        evolution_corr_ps[tense] = np.nan

# Calculer la corrélation entre 'Passé simple' et les autres temps verbaux pour chaque année
for year in evolution_corr_ps['Année']:
    df_year = data[data['Année'] == year]
    for tense in temps_verbaux:
        if tense == 'Passé simple':
            continue  # On ne corrèle pas avec lui-même
        # Calculer la corrélation entre 'Passé simple' et le temps verbal actuel pour l'année en cours
        if df_year[tense].nunique() > 1 and df_year['Passé simple'].nunique() > 1:
            corr = df_year['Passé simple'].corr(df_year[tense])
        else:
            corr = np.nan  # Pas assez de variabilité pour calculer la corrélation
        evolution_corr_ps.loc[evolution_corr_ps['Année'] == year, tense] = corr

# Afficher les premières lignes de l'évolution des corrélations
print("\nÉvolution des corrélations avant lissage :")
print(evolution_corr_ps.head())

# Étape 5 : Appliquer un lissage
window_size = 5  # Taille de la fenêtre pour le lissage

# Appliquer le filtre de lissage à chaque colonne de temps verbal (sauf 'Passé simple')
for tense in temps_verbaux:
    if tense == 'Passé simple':
        continue
    evolution_corr_ps[tense] = uniform_filter1d(evolution_corr_ps[tense].fillna(0), size=window_size)

# Afficher les premières lignes de l'évolution des corrélations après lissage
print("\nÉvolution des corrélations après lissage :")
print(evolution_corr_ps.head())

# Étape 6 : Visualiser l'évolution lissée des corrélations
plt.figure(figsize=(12, 8))

# Tracer chaque courbe de corrélation
for tense in temps_verbaux:
    if tense == 'Passé simple':
        continue
    plt.plot(evolution_corr_ps['Année'], evolution_corr_ps[tense], label=tense, linewidth=2)

# Ajouter une ligne horizontale à y=0 pour référence
plt.axhline(0, color='gray', linestyle='--')

# Personnaliser le graphique
plt.title("Évolution lissée des corrélations entre le Passé Simple et les autres temps verbaux")
plt.xlabel("Année")
plt.ylabel("Coefficient de corrélation")
plt.legend(title="Temps Verbaux")
plt.grid(True)
plt.show()
