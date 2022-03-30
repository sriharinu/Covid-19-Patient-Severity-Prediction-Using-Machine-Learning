import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 2.train test split
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

np.set_printoptions(suppress=True)

covid = pd.read_excel("covid_kaggle.xlsx")

print("Size Of the Dataset before preprocessing is ")
print(covid.shape) 

#  1.DataWash

covid = covid.drop(['Respiratory Syncytial Virus', 'Influenza A', 'Influenza B', 'Parainfluenza 1', 'CoronavirusNL63', 'Parainfluenza 1', 'Chlamydophila pneumoniae', 'Adenovirus', 'Parainfluenza 4', 'Coronavirus229E', 'CoronavirusOC43', 'Inf A H1N1 2009', 'Bordetella pertussis', 'Metapneumovirus', 'Rhinovirus/Enterovirus', 'Coronavirus HKU1', 'Parainfluenza 3', 'Influenza B, rapid test', 'Influenza A, rapid test'], axis=1)
covid = covid.drop(['Patient ID', 'Patient addmited to regular ward (1=yes, 0=no)', 'Patient addmited to semi-intensive unit (1=yes, 0=no)', 'Patient addmited to intensive care unit (1=yes, 0=no)'], axis=1)
urine_features = ['Urine - Esterase', 'Urine - Aspect', 'Urine - pH', 'Urine - Hemoglobin', 'Urine - Bile pigments', 'Urine - Ketone Bodies', 'Urine - Nitrite', 'Urine - Density', 'Urine - Urobilinogen', 'Urine - Protein', 'Urine - Sugar', 'Urine - Leukocytes', 'Urine - Crystals', 'Urine - Red blood cells', 'Urine - Hyaline cylinders', 'Urine - Granular cylinders', 'Urine - Yeasts', 'Urine - Color']
covid = covid.drop(urine_features, axis=1)
arterial_blood_gas_features = ['Hb saturation (arterial blood gases)', 'pCO2 (arterial blood gas analysis)', 'Base excess (arterial blood gas analysis)', 'pH (arterial blood gas analysis)', 'Total CO2 (arterial blood gas analysis)', 'HCO3 (arterial blood gas analysis)', 'pO2 (arterial blood gas analysis)', 'Arteiral Fio2', 'Phosphor', 'ctO2 (arterial blood gas analysis)']
covid = covid.drop(arterial_blood_gas_features, axis=1)

#print(covid.shape)

i = 0
for column in covid:
    if (covid[column].count() < 100):
       # print(column, covid[column].count())
        covid = covid.drop(column, axis=1)

covid = covid.loc[:,covid.apply(pd.Series.nunique) != 1]  

features = list(covid.columns)
sorted_features = [x for _,x in sorted(zip(covid[features].count(), features))]   

covid_init = covid[sorted_features[-1]]
#for i in reversed(range(0, len(sorted_features))):
    #print(sorted_features[i], covid[sorted_features[i]].count())   

removed_features = ['Lactic Dehydrogenase', 'Creatine phosphokinase\xa0(CPK)\xa0', 'International normalized ratio (INR)', 'Base excess (venous blood gas analysis)', 'HCO3 (venous blood gas analysis)', 'Hb saturation (venous blood gas analysis)', 'Total CO2 (venous blood gas analysis)', 'pCO2 (venous blood gas analysis)', 'pH (venous blood gas analysis)', 'pO2 (venous blood gas analysis)', 'Alkaline phosphatase', 'Gamma-glutamyltransferase\xa0', 'Direct Bilirubin', 'Indirect Bilirubin', 'Total Bilirubin', 'Serum Glucose', 'Alanine transaminase', 'Aspartate transaminase', 'Strepto A', 'Sodium', 'Potassium', 'Urea', 'Creatinine']

covid = covid.drop(removed_features, axis=1)

# Drop patients that have less than 10 records

for index, row in covid.iterrows():
    if row.count() < 10:
        covid.drop(index, inplace=True)
        
features = list(covid.columns)
sorted_features = [x for _,x in sorted(zip(covid[features].count(), features))]
#for i in reversed(range(0, len(sorted_features))):
    #print(sorted_features[i], covid[sorted_features[i]].count())
    
    
#Drop NaN

covid = covid.dropna()

# set poitive as 1 and negative as 0
covid['SARS-Cov-2 exam result'] = covid['SARS-Cov-2 exam result'].map({'positive': 1, 'negative': 0})

# we consider 420 rows + 18 Coloumns
print("Size of Dataset after preprocessing is ")
print(covid.shape)

# Test Train Split


y = covid["SARS-Cov-2 exam result"].to_numpy()

X = covid
X = X.drop(["SARS-Cov-2 exam result"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.9, random_state = 1)

X_train.shape

X_test.shape

print(np.sum(y_test), "positive among", len(y_test), "patients in test data")



































