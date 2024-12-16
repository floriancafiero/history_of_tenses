import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# Pour afficher les graphiques directement dans Colab
%matplotlib inline

# Charger les données depuis le fichier CSV
data = pd.read_csv("DF_EVOL_TENSE_FLORIAN.csv")

# Afficher les premières lignes pour vérifier
print("Aperçu des données :")
print(data.head())

# Définir les colonnes correspondant aux temps verbaux
temps_verbaux = ['Présent', 'Imparfait', 'Passé simple', 'Passé composé', 'Futur', 'Plus que parfait']

# Vérifier que toutes les colonnes existent dans les données
for tense in temps_verbaux:
    if tense not in data.columns:
        raise ValueError(f"La colonne '{tense}' n'est pas présente dans les données.")

# Vérifier la présence de la colonne 'Année'
if 'Année' not in data.columns:
    raise ValueError("La colonne 'Année' n'est pas présente dans les données.")

# Trier les données par année pour assurer une progression chronologique
data = data.sort_values('Année')

# Calculer la matrice de corrélation
correlation_matrix = data[temps_verbaux].corr()

# Afficher la matrice de corrélation
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Matrice de corrélation entre les temps verbaux")
plt.show()

# Initialiser un DataFrame pour stocker les corrélations
evolution_corr_ps = pd.DataFrame()
evolution_corr_ps['Année'] = data['Année'].unique()

# Calculer la corrélation entre 'Passé simple' et les autres temps verbaux pour chaque année
for tense in temps_verbaux:
    if tense == 'Passé simple':
        continue  # On ne corrèle pas avec lui-même
    corr_values = []
    for year in evolution_corr_ps['Année']:
        # Filtrer les données pour l'année en cours
        df_year = data[data['Année'] == year]
        if df_year.empty:
            corr = 0
        else:
            # Calculer la corrélation entre 'Passé simple' et le temps verbal actuel
            corr = df_year['Passé simple'].corr(df_year[tense])
            # Gérer les valeurs NaN résultant de la corrélation
            corr = corr if pd.notnull(corr) else 0
        corr_values.append(corr)
    evolution_corr_ps[tense] = corr_values

# Afficher les premières lignes de l'évolution des corrélations
print("\nÉvolution des corrélations avant lissage :")
print(evolution_corr_ps.head())

# Appliquer le filtre de lissage à chaque colonne de temps verbal (sauf 'Année')
window_size = 5  # Taille de la fenêtre pour le lissage
for col in evolution_corr_ps.columns[1:]:
    evolution_corr_ps[col] = uniform_filter1d(evolution_corr_ps[col], size=window_size)

# Afficher les premières lignes de l'évolution des corrélations après lissage
print("\nÉvolution des corrélations après lissage :")
print(evolution_corr_ps.head())

# Tracer l'évolution lissée des corrélations
plt.figure(figsize=(12, 8))

# Tracer chaque courbe de corrélation
for col in evolution_corr_ps.columns[1:]:
    plt.plot(evolution_corr_ps['Année'], evolution_corr_ps[col], label=col, linewidth=2)

# Ajouter une ligne horizontale à y=0 pour référence
plt.axhline(0, color='gray', linestyle='--')

# Personnaliser le graphique
plt.title("Évolution lissée des corrélations entre le Passé Simple et les autres temps verbaux")
plt.xlabel("Année")
plt.ylabel("Coefficient de corrélation")
plt.legend(title="Temps Verbaux")
plt.grid(True)
plt.show()
