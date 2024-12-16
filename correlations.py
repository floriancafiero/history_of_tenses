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

# Evolution de corrélations entre passé simple et autres temps
window_size = 5  
for col in evolution_corr_ps.columns[1:]:  # Sauf la colonne "Année"
    evolution_corr_ps[col] = uniform_filter1d(evolution_corr_ps[col], size=window_size)


plt.figure(figsize=(12, 8))
for col in evolution_corr_ps.columns[1:]:  # Ignorer "Année"
    plt.plot(evolution_corr_ps['Année'], evolution_corr_ps[col], label=col.split('_')[-1], linewidth=2)

plt.axhline(0, color='gray', linestyle='--')
plt.title("Évolution lissée des corrélations entre le Passé Simple et les autres temps verbaux")
plt.xlabel("Année")
plt.ylabel("Coefficient de corrélation")
plt.legend(title="Temps Verbaux")
plt.grid(True)
plt.show()
