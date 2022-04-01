import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

################################################################################################## 2.Test Train Split


y = covid["SARS-Cov-2 exam result"].to_numpy()

X = covid
X = X.drop(["SARS-Cov-2 exam result"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.9, random_state = 1)

X_train.shape

X_test.shape

print(np.sum(y_test), "positive among", len(y_test), "patients in test data")


###################################################################################################  3.Feature Selection

from sklearn.feature_selection import VarianceThreshold

def drop_features(X_train, X_test, threshhold):
    sel = VarianceThreshold(threshold=threshhold)
    sel.fit(X_train)
    # print("No. of constant features:",
    #     len([
    #         x for x in X_train.columns
    #         if x not in X_train.columns[sel.get_support()]
    #     ])
    # )
    constant_features = [x for x in X_train.columns if x not in X_train.columns[sel.get_support()]]

    #print(constant_features)
    X_train.drop(labels=constant_features, axis=1, inplace=True)
    X_test.drop(labels=constant_features, axis=1, inplace=True)
    
drop_features(X_train, X_test, 0.01)

covid_t = covid.T
# print("No. of Duplicated Features:", covid_t.duplicated().sum())
# print(covid_t[covid_t.duplicated()].index.values)

#Correlation

corrmat = X_train.corr()
corrmat = corrmat.abs().unstack()
corrmat = corrmat.sort_values(ascending=False)
corrmat = corrmat[corrmat >= 0.8]
corrmat = corrmat[corrmat < 1]
corrmat = pd.DataFrame(corrmat).reset_index()
corrmat.columns = ['feature1', 'feature2', 'corr']
corrmat


# find groups of correlated features

grouped_feature_ls = []
correlated_groups = []

for feature in corrmat.feature1.unique():
    if feature not in grouped_feature_ls:

        # find all features correlated to a single feature
        correlated_block = corrmat[corrmat.feature1 == feature]
        grouped_feature_ls = grouped_feature_ls + list(
            correlated_block.feature2.unique()) + [feature]

        # append the block of features to the list
        correlated_groups.append(correlated_block)

#print('found {} correlated groups'.format(len(correlated_groups)))
#print('out of {} total features'.format(X_train.shape[1]))
    
    
# now we can visualise each group. We see that some groups contain
# only 2 correlated features, some other groups present several features 
# that are correlated among themselves.

# for group in correlated_groups:
#     print(group)
#     print()


def correlation(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j] >= threshold):
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr


corr_features = list((correlation(X_train, 0.8)))
#print(corr_features)

X_train.drop(labels=corr_features, axis=1, inplace=True)
X_test.drop(labels=corr_features, axis=1, inplace=True)

X_train.shape, X_test.shape

###3.3 Statistical Methods

from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.feature_selection import SelectKBest, SelectPercentile

mi = mutual_info_classif(X_train, y_train)
mi = pd.Series(mi)
mi.index = X_train.columns

mi.sort_values(ascending=False).plot.bar(figsize=(20, 8))

sel_ = SelectKBest(mutual_info_classif, k = 10).fit(X_train, y_train)

mi_features = list(X_train.columns[ ~ sel_.get_support()].values)

mi_features

X_train.drop(labels=mi_features, axis=1, inplace=True)
X_test.drop(labels=mi_features, axis=1, inplace=True)

X_train.shape

X_test.shape


#################################################################################3. Classifier

import sklearn
import sklearn.ensemble
import sklearn.metrics
import xgboost as xgb

from sklearn.model_selection import cross_val_score

def cv_score(classifier, X, y, scoring):
    return cross_val_score(classifier, X, y, cv=5, scoring=scoring)

# 3.1 Decision Tree

dt = sklearn.tree.DecisionTreeClassifier()

dt_f1 = cv_score(dt, X_train, y_train, 'f1')

dt.fit(X_train, y_train)

print(np.mean(dt_f1))

dt_pred = dt.predict(X_test)

print("Decision Tree")
print("Precision: ", sklearn.metrics.accuracy_score(y_test, dt_pred))
# print("Recal: ", sklearn.metrics.recall_score(y_test, dt_pred))
# print("F1: ", sklearn.metrics.f1_score(y_test, dt_pred))

#print('Prediction:', ' '.join(str(e) for e in dt_pred))
#print('     Truth:', ' '.join(str(e) for e in y_test))

#  3.2 Random Forests




























