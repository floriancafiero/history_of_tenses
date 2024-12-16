import seaborn as sns
import matplotlib.pyplot as plt

# Calculer la matrice de corrélation entre les temps verbaux
temps_verbaux = ['Présent', 'Imparfait', 'Passé simple', 'Passé composé', 'Futur', 'Plus que parfait']
correlation_matrix = data[temps_verbaux].corr()

# Visualisation de la matrice de corrélation
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Matrice de corrélation entre les temps verbaux")
plt.show()
