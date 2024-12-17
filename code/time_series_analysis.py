# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Load the data
file_path = 'DF_EVOL_TENSE_FLORIAN.csv' 
data = pd.read_csv(file_path)

# Preprocess the data
# Convert 'date' column to datetime format and extract the year
data['date'] = pd.to_datetime(data['date'])
data['year'] = data['date'].dt.year

# List of tense columns to analyze
tense_columns = ['Présent', 'Imparfait', 'Passé simple', 'Passé composé', 'Futur', 'Plus que parfait']

# Calculate percentage for each tense
data['total_tenses'] = data[tense_columns].sum(axis=1)
for tense in tense_columns:
    data[tense + '_pct'] = (data[tense] / data['total_tenses']) * 100

# Group by year and average the percentages
grouped_pct_data = data.groupby('year')[[tense + '_pct' for tense in tense_columns]].mean()

# Handle missing values (interpolate where needed)
grouped_pct_data = grouped_pct_data.interpolate()

# Define a function for time series decomposition
def decompose_time_series(df, column):
    """
    Decompose time series into trend, seasonal, and residual components.
    
    Parameters:
    df: DataFrame with the time series data.
    column: Name of the column to decompose.
    """
    # Perform decomposition
    result = seasonal_decompose(df[column], model='additive', period=10, extrapolate_trend='freq')

    # Plot decomposition
    plt.figure(figsize=(12, 8))
    plt.subplot(411)
    plt.plot(df.index, df[column], label='Original', color='blue')
    plt.legend(loc='upper left')
    plt.subplot(412)
    plt.plot(result.trend, label='Trend', color='orange')
    plt.legend(loc='upper left')
    plt.subplot(413)
    plt.plot(result.seasonal, label='Seasonal', color='green')
    plt.legend(loc='upper left')
    plt.subplot(414)
    plt.plot(result.resid, label='Residual', color='red')
    plt.legend(loc='upper left')
    plt.suptitle(f'Time Series Decomposition for {column}', fontsize=14)
    plt.tight_layout()
    plt.show()

# Apply decomposition for all tenses
for tense in tense_columns:
    decompose_time_series(grouped_pct_data, tense + '_pct')
