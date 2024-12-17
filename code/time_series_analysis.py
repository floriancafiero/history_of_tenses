import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import unicodedata
import re

def clean_string_value(value):
    if pd.isnull(value):
        return value
    nfkd_form = unicodedata.normalize('NFKD', value)
    only_ascii = nfkd_form.encode('ASCII', 'ignore').decode('utf-8')
    only_ascii = re.sub(r"[^\w\s-]", "", only_ascii)
    only_ascii = only_ascii.strip().replace(' ', '_').lower()
    return only_ascii

def clean_column_names(col_names):
    cleaned = []
    for name in col_names:
        cleaned.append(clean_string_value(name))
    return cleaned

df = pd.read_csv('/content/df_metadated_floflo_updated.csv', delimiter=',', encoding='utf-8')
df.columns = clean_column_names(df.columns)
print("Cleaned Column Names:")
print(df.columns)

required_columns = [
    'col_name', 'present', 'imparfait', 'passe_simple', 
    'passe_compose', 'futur', 'plus_que_parfait', 'date', 
    'canon', 'subgenre'
]

missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print("\nWarning: Missing columns:", missing_columns)
else:
    print("\nAll required columns are present.")

df['total_verbes'] = df[['present', 'imparfait', 'passe_simple', 'passe_compose', 'futur', 'plus_que_parfait']].sum(axis=1)
df['proportion_present'] = df['present'] / df['total_verbes']
df['proportion_passe_simple'] = df['passe_simple'] / df['total_verbes']

def assign_period_21(year):
    if 1811 <= year <= 1831:
        return '1811_1831'
    elif 1832 <= year <= 1852:
        return '1832_1852'
    elif 1853 <= year <= 1873:
        return '1853_1873'
    elif 1874 <= year <= 1894:
        return '1874_1894'
    elif 1895 <= year <= 1915:
        return '1895_1915'
    elif 1916 <= year <= 1936:
        return '1916_1936'
    elif 1937 <= year <= 1957:
        return '1937_1957'
    elif 1958 <= year <= 1978:
        return '1958_1978'
    elif 1979 <= year <= 1999:
        return '1979_1999'
    elif 2000 <= year <= 2024:
        return '2000_2024'
    else:
        return np.nan

df['period_21'] = df['date'].apply(assign_period_21)
df = df.dropna(subset=['period_21'])

df['subgenre'] = df['subgenre'].astype(str).apply(clean_string_value)
df = df[~df['subgenre'].isna() & (df['subgenre']!='')]

df['subgenre'] = df['subgenre'].replace({'cycles_and_series':'cycles_and_series'})
if 'cycles_and_series' not in df['subgenre'].unique():
    df['subgenre'] = df['subgenre'].replace({'cycles_and_serie':'cycles_and_series'})
df['subgenre'] = pd.Categorical(
    df['subgenre'],
    categories=['cycles_and_series','adventure_novel','children_literature','detective_novel','epistolary_novel','eroticism','fantasy','historical_novel','memoirs_and_autobiography','romantic_novel','science_fiction','short_stories','travel_narrative'],
    ordered=False
)
df = df[df['subgenre'].notna()]

df['period_21'] = pd.Categorical(
    df['period_21'],
    categories=['1811_1831','1832_1852','1853_1873','1874_1894','1895_1915','1916_1936','1937_1957','1958_1978','1979_1999','2000_2024'],
    ordered=True
)

df['canon'] = df['canon'].astype(bool)
epsilon = 1e-5
df['proportion_present'] = df['proportion_present'].clip(epsilon, 1 - epsilon)
df['proportion_passe_simple'] = df['proportion_passe_simple'].clip(epsilon, 1 - epsilon)

subgenre_dummies = pd.get_dummies(df['subgenre'], prefix='', prefix_sep='', drop_first=True)
subgenre_dummies.columns = [col for col in subgenre_dummies.columns]

period_dummies = pd.get_dummies(df['period_21'], prefix='', prefix_sep='', drop_first=True)
# Rename period dummy columns to start with a letter for Patsy
period_map = {}
for c in period_dummies.columns:
    new_c = 'p_' + c
    period_map[c] = new_c
period_dummies.rename(columns=period_map, inplace=True)

df = pd.concat([df, subgenre_dummies, period_dummies], axis=1)

print("\nReference categories:")
print("Reference period: 1811_1831")
print("Reference subgenre: cycles_and_series")

# OLS for proportion of present
formula_reg1 = 'proportion_present ~ canon'
for pcol in period_dummies.columns:
    formula_reg1 += ' + ' + pcol
for scol in subgenre_dummies.columns:
    formula_reg1 += ' + ' + scol

print("\nRunning Regression 1: Proportion of Present")
model_reg1 = smf.ols(formula=formula_reg1, data=df).fit()
print("\nRegression 1 Summary:")
print(model_reg1.summary())

params_reg1 = model_reg1.params
conf_reg1 = model_reg1.conf_int()
conf_reg1.columns = ['2.5%', '97.5%']
coef_reg1 = pd.concat([conf_reg1, params_reg1], axis=1)
coef_reg1.columns = ['2.5%', '97.5%', 'Coefficient']
coef_reg1_no_intercept = coef_reg1.drop('Intercept', errors='ignore')

labels_reg1 = coef_reg1_no_intercept.index.tolist()
new_labels_reg1 = []
for lab in labels_reg1:
    new_lab = lab
    new_lab = new_lab.replace('canon[T.True]', 'canon')
    new_lab = new_lab.replace('canon', 'canon')
    new_lab = new_lab.replace('p_', '')
    new_lab = new_lab.replace('_literature', '_literature')
    new_lab = new_lab.replace('_novel', '_novel')
    new_lab = new_lab.replace('_and_', '_and_')
    new_lab = new_lab.replace('_narrative', '_narrative')
    new_lab = new_lab.replace('children_literature', 'children_literature')
    new_lab = new_lab.replace('adventure_novel', 'adventure_novel')
    new_lab = new_lab.replace('detective_novel', 'detective_novel')
    new_lab = new_lab.replace('epistolary_novel', 'epistolary_novel')
    new_lab = new_lab.replace('eroticism', 'eroticism')
    new_lab = new_lab.replace('fantasy', 'fantasy')
    new_lab = new_lab.replace('historical_novel', 'historical_novel')
    new_lab = new_lab.replace('memoirs_and_autobiography', 'memoirs_and_autobiography')
    new_lab = new_lab.replace('romantic_novel', 'romantic_novel')
    new_lab = new_lab.replace('science_fiction', 'science_fiction')
    new_lab = new_lab.replace('short_stories', 'short_stories')
    new_lab = new_lab.replace('travel_narrative', 'travel_narrative')
    new_labels_reg1.append(new_lab)

colors_reg1 = []
for i, lab in enumerate(labels_reg1):
    if 'canon' in lab:
        colors_reg1.append('orange')
    elif lab.startswith('p_'):
        colors_reg1.append('blue')
    else:
        colors_reg1.append('red')

plt.figure(figsize=(12, 8))
plt.barh(new_labels_reg1,
         coef_reg1_no_intercept['Coefficient'],
         xerr=[coef_reg1_no_intercept['Coefficient'] - coef_reg1_no_intercept['2.5%'], 
               coef_reg1_no_intercept['97.5%'] - coef_reg1_no_intercept['Coefficient']],
         capsize=4, color=colors_reg1)
plt.axvline(x=0, color='black', linestyle='--')
plt.title('Regression 1: Coefficients for Proportion of Present')
plt.xlabel('Coefficient Value')
plt.tight_layout()
plt.show()

# OLS for proportion of passé simple
formula_reg2 = 'proportion_passe_simple ~ canon'
for pcol in period_dummies.columns:
    formula_reg2 += ' + ' + pcol
for scol in subgenre_dummies.columns:
    formula_reg2 += ' + ' + scol

print("\nRunning Regression 2: Proportion of Passé Simple")
model_reg2 = smf.ols(formula=formula_reg2, data=df).fit()
print("\nRegression 2 Summary:")
print(model_reg2.summary())

params_reg2 = model_reg2.params
conf_reg2 = model_reg2.conf_int()
conf_reg2.columns = ['2.5%', '97.5%']
coef_reg2 = pd.concat([conf_reg2, params_reg2], axis=1)
coef_reg2.columns = ['2.5%', '97.5%', 'Coefficient']
coef_reg2_no_intercept = coef_reg2.drop('Intercept', errors='ignore')

labels_reg2 = coef_reg2_no_intercept.index.tolist()
new_labels_reg2 = []
for lab in labels_reg2:
    new_lab = lab
    new_lab = new_lab.replace('canon[T.True]', 'canon')
    new_lab = new_lab.replace('canon', 'canon')
    new_lab = new_lab.replace('p_', '')
    new_lab = new_lab.replace('_literature', '_literature')
    new_lab = new_lab.replace('_novel', '_novel')
    new_lab = new_lab.replace('_and_', '_and_')
    new_lab = new_lab.replace('_narrative', '_narrative')
    new_lab = new_lab.replace('children_literature', 'children_literature')
    new_lab = new_lab.replace('adventure_novel', 'adventure_novel')
    new_lab = new_lab.replace('detective_novel', 'detective_novel')
    new_lab = new_lab.replace('epistolary_novel', 'epistolary_novel')
    new_lab = new_lab.replace('eroticism', 'eroticism')
    new_lab = new_lab.replace('fantasy', 'fantasy')
    new_lab = new_lab.replace('historical_novel', 'historical_novel')
    new_lab = new_lab.replace('memoirs_and_autobiography', 'memoirs_and_autobiography')
    new_lab = new_lab.replace('romantic_novel', 'romantic_novel')
    new_lab = new_lab.replace('science_fiction', 'science_fiction')
    new_lab = new_lab.replace('short_stories', 'short_stories')
    new_lab = new_lab.replace('travel_narrative', 'travel_narrative')
    new_labels_reg2.append(new_lab)

colors_reg2 = []
for i, lab in enumerate(labels_reg2):
    if 'canon' in lab:
        colors_reg2.append('orange')
    elif lab.startswith('p_'):
        colors_reg2.append('blue')
    else:
        colors_reg2.append('red')

plt.figure(figsize=(12, 8))
plt.barh(new_labels_reg2,
         coef_reg2_no_intercept['Coefficient'],
         xerr=[coef_reg2_no_intercept['Coefficient'] - coef_reg2_no_intercept['2.5%'], 
               coef_reg2_no_intercept['97.5%'] - coef_reg2_no_intercept['Coefficient']],
         capsize=4, color=colors_reg2)
plt.axvline(x=0, color='black', linestyle='--')
plt.title('Regression 2: Coefficients for Proportion of Passé Simple')
plt.xlabel('Coefficient Value')
plt.tight_layout()
plt.show()
