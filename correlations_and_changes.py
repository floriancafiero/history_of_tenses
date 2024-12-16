import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from matplotlib.lines import Line2D
import warnings

%matplotlib inline
warnings.filterwarnings('ignore')

# Load the dataset
data = pd.read_csv("DF_EVOL_TENSE_FLORIAN.csv")
print("Data Preview:")
print(data.head())

# Convert 'date' to datetime format and extract 'Year'
data['date'] = pd.to_datetime(data['date'], errors='coerce')
data['Year'] = data['date'].dt.year

# Drop rows with invalid dates
missing_years = data['Year'].isnull().sum()
if missing_years > 0:
    print(f"\nWarning: {missing_years} entries have invalid dates and will be removed.")
    data = data.dropna(subset=['Year'])

data['Year'] = data['Year'].astype(int)
print("\nAvailable Years in Data:", sorted(data['Year'].unique()))

# Define verb tense columns and their English translations
verb_tenses = ['Présent', 'Imparfait', 'Passé simple', 'Passé composé', 'Futur', 'Plus que parfait']
for tense in verb_tenses:
    if tense not in data.columns:
        raise ValueError(f"Column '{tense}' is missing from the data.")

tense_translations = {
    'Présent': 'Present',
    'Imparfait': 'Imperfect',
    'Passé simple': 'Simple Past',
    'Passé composé': 'Past Perfect',
    'Futur': 'Future',
    'Plus que parfait': 'Pluperfect'
}

# Compute and plot correlation matrix
correlation_matrix = data[verb_tenses].corr()
correlation_matrix_english = correlation_matrix.rename(columns=tense_translations, index=tense_translations)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_english, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix Between Verb Tenses", fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(rotation=0, fontsize=12)
plt.tight_layout()
plt.show()

# Calculate yearly correlations
yearly_corr = pd.DataFrame()
yearly_corr['Year'] = sorted(data['Year'].unique())

for tense in verb_tenses:
    if tense != 'Passé simple':
        yearly_corr[tense_translations[tense]] = np.nan

for year in yearly_corr['Year']:
    df_year = data[data['Year'] == year]
    for tense in verb_tenses:
        if tense == 'Passé simple':
            continue
        if df_year[tense].nunique() > 1 and df_year['Passé simple'].nunique() > 1:
            corr = df_year['Passé simple'].corr(df_year[tense])
        else:
            corr = np.nan
        yearly_corr.loc[yearly_corr['Year'] == year, tense_translations[tense]] = corr

print("\nYearly Correlations Before Smoothing:")
print(yearly_corr.head())

# Apply increased smoothing
window_size = 7
for tense in tense_translations.values():
    if tense == 'Simple Past':
        continue
    yearly_corr[tense] = uniform_filter1d(yearly_corr[tense].fillna(0), size=window_size)

print("\nYearly Correlations After Smoothing:")
print(yearly_corr.head())

# Rename correlation columns to include '_corr'
for tense in verb_tenses:
    if tense != 'Passé simple':
        english_tense = tense_translations[tense]
        yearly_corr.rename(columns={english_tense: f'{english_tense}_corr'}, inplace=True)

# Define color groups
group1_colors = sns.color_palette("Blues", n_colors=3)
group2_colors = sns.color_palette("Oranges", n_colors=3)

group1 = ['Past Perfect', 'Present', 'Future']
group2 = ['Imperfect', 'Simple Past' , 'Pluperfect']

color_mapping = {}
for tense, color in zip(group1, group1_colors):
    color_mapping[tense] = color
for tense, color in zip(group2, group2_colors):
    color_mapping[tense] = color

# Define enhanced plotting function
def plot_smoothed_correlations(data, tenses, color_mapping, title):
    plt.figure(figsize=(14, 8))
    for tense in tenses:
        english_tense = tense
        perc_col = f'{english_tense}_corr'
        if perc_col not in data.columns:
            continue
        plt.plot(
            data['Year'],
            data[perc_col],
            label=english_tense,
            color=color_mapping.get(english_tense, 'gray'),
            linewidth=2,
            alpha=0.7
        )
    plt.title(title, fontsize=16)
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("Correlation Coefficient", fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title="Verb Tenses", fontsize=12, title_fontsize=14, loc='best')
    plt.tight_layout()
    plt.show()

# Plot smoothed correlations
plot_smoothed_correlations(
    data=yearly_corr,
    tenses=group1 + group2,
    color_mapping=color_mapping,
    title="Smoothed Evolution of Correlations Between 'Simple Past' and Other Verb Tenses"
)

# Aggregate data by year and calculate percentages
aggregated_data = data.groupby('Year')[verb_tenses].sum().reset_index()
aggregated_data['Total'] = aggregated_data[verb_tenses].sum(axis=1)

zero_total_years = aggregated_data[aggregated_data['Total'] == 0]
if not zero_total_years.empty:
    print(f"\nWarning: {len(zero_total_years)} years have a total usage of zero and will be removed.")
    aggregated_data = aggregated_data[aggregated_data['Total'] != 0]

for tense in verb_tenses:
    english_tense = tense_translations[tense]
    aggregated_data[f'{english_tense}_perc'] = (aggregated_data[tense] / aggregated_data['Total']) * 100

percentage_columns = [f'{tense_translations[tense]}_perc' for tense in verb_tenses]
aggregated_data.dropna(subset=percentage_columns, inplace=True)

print("\nPercentage Usage of Verb Tenses per Year:")
print(aggregated_data.head())

# Define verb groups
group1 = ['Simple Past', 'Imperfect', 'Pluperfect']
group2 = ['Present', 'Future', 'Past Perfect']

# Create percentage columns for each group
group1_perc = [f'{tense}_perc' for tense in group1]
group2_perc = [f'{tense}_perc' for tense in group2]

# Function to detect anomalies using Isolation Forest
def detect_anomalies_isolation_forest(series, contamination=0.01, random_state=42):
    data_model = series.values.reshape(-1, 1)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_model)
    model = IsolationForest(contamination=contamination, random_state=random_state, n_estimators=100)
    model.fit(data_scaled)
    preds = model.predict(data_scaled)
    anomalies = np.where(preds == -1)[0]
    return anomalies

# Detect anomalies
anomalies_group1 = {}
anomalies_group2 = {}

for tense in group1:
    perc_col = f'{tense}_perc'
    anomalies = detect_anomalies_isolation_forest(aggregated_data[perc_col], contamination=0.01)
    anomalies_group1[tense] = anomalies

for tense in group2:
    perc_col = f'{tense}_perc'
    anomalies = detect_anomalies_isolation_forest(aggregated_data[perc_col], contamination=0.01)
    anomalies_group2[tense] = anomalies

print("\nDetected Anomalies for Group 1:")
for tense, anomalies in anomalies_group1.items():
    anomaly_years = aggregated_data['Year'].iloc[anomalies].tolist()
    print(f"{tense}: {len(anomalies)} anomalies at years {anomaly_years}")

print("\nDetected Anomalies for Group 2:")
for tense, anomalies in anomalies_group2.items():
    anomaly_years = aggregated_data['Year'].iloc[anomalies].tolist()
    print(f"{tense}: {len(anomalies)} anomalies at years {anomaly_years}")

# Enhanced plotting function for usage with anomalies
def plot_usage_with_anomalies_fixed(data, tenses, anomalies_dict, title):
    plt.figure(figsize=(14, 8))
    for tense in tenses:
        perc_col = f'{tense}_perc'
        line, = plt.plot(data['Year'], data[perc_col], label=tense, linewidth=2)
        anomalies = anomalies_dict.get(tense, [])
        if len(anomalies) > 0:
            anomaly_years = data['Year'].iloc[anomalies]
            anomaly_values = data[perc_col].iloc[anomalies]
            plt.scatter(anomaly_years, anomaly_values, color=line.get_color(), marker='o', s=100, label='_nolegend_')
    anomaly_marker = Line2D([0], [0], marker='o', color='w', label='Anomaly',
                            markerfacecolor='black', markersize=10)
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(anomaly_marker)
    labels.append('Anomaly')
    plt.legend(handles, labels, title="Verb Tenses and Anomalies", loc='upper right')
    plt.title(title, fontsize=16)
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("Usage Percentage (%)", fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.show()

# Plot usage percentages with anomalies
plot_usage_with_anomalies_fixed(
    data=aggregated_data,
    tenses=group1,
    anomalies_dict=anomalies_group1,
    title="Usage Percentage of Verb Tenses: Simple Past, Imperfect, Pluperfect with Anomalies"
)

plot_usage_with_anomalies_fixed(
    data=aggregated_data,
    tenses=group2,
    anomalies_dict=anomalies_group2,
    title="Usage Percentage of Verb Tenses: Present, Future, Past Perfect with Anomalies"
)
