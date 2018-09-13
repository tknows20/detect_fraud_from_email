#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib
matplotlib.use('QT4Agg')
import matplotlib.pyplot as plt
from scipy import stats

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier

### "Task 1: Select what features you'll use."
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','total_payments','total_stock_value', 'fraction_from_poi', 'fraction_to_poi', 'shared_receipt_with_poi'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### "Task 2: Remove outliers"
data_dict.pop("TOTAL", 0)

print "=================================================="
print "===== Preliminary exploratory data analysis ======"
print "=================================================="

### Import into a dataframe for easy manipulation 
df = pd.DataFrame.from_dict(data_dict,orient='index')
df = df.where(df!='NaN',None)
df = df.apply(pd.to_numeric, errors = 'ignore')
print ""
print "===== Data Set Info ====="
df.info()

df.hist(figsize = (20,15));
plt.suptitle("Histogram of All Numerical Data")
plt.show()

print ""
print "===== Data Set Statistics ====="
print df.describe(include = 'all')

### "Task 3: Create new feature(s)"
df_new = df[['poi','total_payments','total_stock_value','to_messages','from_messages','from_this_person_to_poi','from_poi_to_this_person','shared_receipt_with_poi']]
df_new = df_new.assign(fraction_from_poi = 1.0*df_new.from_poi_to_this_person / df_new.to_messages)
df_new = df_new.assign(fraction_to_poi = 1.0*df_new.from_this_person_to_poi / df_new.from_messages)

### Plot scatter matrix as first check for collinearity 
df_emails = df_new[['from_poi_to_this_person','from_this_person_to_poi','to_messages','from_messages','fraction_from_poi', 'fraction_to_poi','shared_receipt_with_poi']]
df_payments = df[['salary','bonus','long_term_incentive','deferred_income','deferral_payments','loan_advances','other', 'expenses', 'director_fees','total_payments']]
df_stocks = df[['exercised_stock_options','restricted_stock','restricted_stock_deferred','total_stock_value']]

df_payments = df_payments.fillna(0)
df_stocks = df_stocks.fillna(0)
df_emails = df_emails.fillna(0)
df_pcheck = df_payments[(np.abs(stats.zscore(df_payments)) < 3).all(axis=1)]
df_scheck = df_stocks[(np.abs(stats.zscore(df_stocks)) < 3).all(axis=1)]
df_echeck = df_emails[(np.abs(stats.zscore(df_emails)) < 3).all(axis=1)]

scatter_matrix(df_pcheck,figsize = (20,15));
plt.suptitle("Scatter Plots Comparing Payment-Related Data")
plt.show()
scatter_matrix(df_scheck,figsize = (20,15)); 
plt.suptitle("Scatter Plots Comparing Stock-Related Data")
plt.show()
scatter_matrix(df_echeck,figsize = (20,15));
plt.suptitle("Scatter Plots Comparing Email Data")
plt.show()

### Calculate Variance Inflation Factor  as second check for collinearity
print ""
print "===== VIF for Payment-Related Information ====="
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(df_payments.values, i) for i in range(df_payments.shape[1])]
vif["features"] = df_payments.columns
print vif.round(1)

print ""
print "===== VIF for Stock-Related Information ====="
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(df_stocks.values, i) for i in range(df_stocks.shape[1])]
vif["features"] = df_stocks.columns
print vif.round(1)

print ""
print "===== VIF for Email-Related Information ====="
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(df_emails.values, i) for i in range(df_emails.shape[1])]
vif["features"] = df_emails.columns
print vif.round(1)


### Box and Whisker Plots to Compare POI and non-POI data
df_new_fin = df_new[['poi','total_payments','total_stock_value']]
df_new_email = df_new[['poi','from_poi_to_this_person','from_messages','from_this_person_to_poi','to_messages', 'shared_receipt_with_poi']]
df_new_email_ratio = df_new[['poi','fraction_from_poi','fraction_to_poi']]

df_new_fin.boxplot(by = 'poi', layout=(2,1), figsize=(20,10));
plt.suptitle("Comparing Payment and Stock Related Data Between POI and non-POI")
plt.ylim(0,0.4e8); 
plt.show();

df_new_email.boxplot(by = 'poi', layout=(5,1), figsize=(20,50));
plt.suptitle("Comparing Email Related Data Between POI and non-POI")
plt.ylim(0,4e3);
plt.show();

df_new_email_ratio.boxplot(by = 'poi', layout=(2,1), figsize=(20,50));
plt.suptitle("Comparing Email Related Ratios Between POI and non-POI")
plt.show();


### Store to my_dataset for easy export below.
df_new = df_new.fillna(0)
data_dict = df_new.to_dict('index')
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### "Task 4: Try a varity of classifiers"
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

### Feature Scaling (emails and dollars have different units)
scaler_std = StandardScaler()
features_scaled = scaler_std.fit_transform(features)

### Do k-fold validation to compare classifiers 
### Do parameter GridSearchCV within loop to get best parameters

import warnings
warnings.filterwarnings('ignore')

print ""
print "===== Looping through k-folds Validation ====="

kf = KFold(n_splits=3, shuffle = True, random_state=42)

metric_scoring_NB = [0,0,0,0]
metric_scoring_SVM = [0,0,0,0]
metric_scoring_DT = [0,0,0,0]
metric_scoring_kNN = [0,0,0,0]
metric_scoring_AB = [0,0,0,0]
metric_scoring_RF = [0,0,0,0]
metric_scoring_LR = [0,0,0,0]

for train_idx, test_idx in kf.split(labels): 
    features_train = []
    features_test  = []
    labels_train   = []
    labels_test    = []
    
    for ii in train_idx:
        features_train.append( features_scaled[ii] )
        labels_train.append( labels[ii] )
    for jj in test_idx:
        features_test.append( features_scaled[jj] )
        labels_test.append( labels[jj] )

    ### 1. GAUSSIAN
    clf = GaussianNB()
    clf.fit(features_train, labels_train)
    predictions = clf.predict(features_test)
        
    metric_scoring_NB[0] += clf.score(features_test, labels_test)
    metric_scoring_NB[1] += precision_score(labels_test, predictions)
    metric_scoring_NB[2] += recall_score(labels_test, predictions)
    metric_scoring_NB[3] += f1_score(labels_test, predictions)
    
    ### 2. SVM
    param_grid = {
         'C': [1e-2, 1e-1, 1, 10, 100, 1000],
          'gamma': [1e-2, 1e-1, 1, 10, 100, 1000]
    }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid, scoring = 'f1');
    clf.fit(features_train, labels_train)
    predictions = clf.predict(features_test)
    print "SVM Parameters: ", clf.best_estimator_
        
    metric_scoring_SVM[0] += clf.score(features_test, labels_test)
    metric_scoring_SVM[1] += precision_score(labels_test, predictions)
    metric_scoring_SVM[2] += recall_score(labels_test, predictions)
    metric_scoring_SVM[3] += f1_score(labels_test, predictions)
    
    ### 3. Decision Tree
    param_grid = {
    'min_samples_split': [2,3,4,5,10,15,20,25,30,35,40]
    }
    clf = GridSearchCV(DecisionTreeClassifier(),param_grid, scoring =  'f1');
    clf.fit(features_train, labels_train)
    predictions = clf.predict(features_test)
    print "Decision Tree Parameters: ", clf.best_estimator_
        
    metric_scoring_DT[0] += clf.score(features_test, labels_test)
    metric_scoring_DT[1] += precision_score(labels_test, predictions)
    metric_scoring_DT[2] += recall_score(labels_test, predictions)
    metric_scoring_DT[3] += f1_score(labels_test, predictions)
    
    ### 4. k Nearest Neighbors
    param_grid = {
    'n_neighbors': [2, 3, 4, 5],
    'weights' : ['uniform','distance']
    }
    clf = GridSearchCV(KNeighborsClassifier(),param_grid, scoring =  'f1');
    clf.fit(features_train, labels_train)
    predictions = clf.predict(features_test)
    print "kNN Parameters: ", clf.best_estimator_
        
    metric_scoring_kNN[0] += clf.score(features_test, labels_test)
    metric_scoring_kNN[1] += precision_score(labels_test, predictions)
    metric_scoring_kNN[2] += recall_score(labels_test, predictions)
    metric_scoring_kNN[3] += f1_score(labels_test, predictions)
        
    ### 5. AdaBoost
    param_grid = {
    'n_estimators': [1,2,3,4,5,6,7],
    'learning_rate' : [1e-4,1e-3,1e-2,0.1,1,10,100]
    }
    clf = GridSearchCV(AdaBoostClassifier(), param_grid, scoring =  'f1');
    clf.fit(features_train, labels_train)
    predictions = clf.predict(features_test)
    print "AdaBoost Parameters: ", clf.best_estimator_
        
    metric_scoring_AB[0] += clf.score(features_test, labels_test)
    metric_scoring_AB[1] += precision_score(labels_test, predictions)
    metric_scoring_AB[2] += recall_score(labels_test, predictions)
    metric_scoring_AB[3] += f1_score(labels_test, predictions)
  
    ### 6. Random Forest
    param_grid = {
    'n_estimators': [1,5,10,15,20,25,30],
    'min_samples_split':[2,4,6,8,10,15,20,25,30,40]
    }
    clf = GridSearchCV(RandomForestClassifier(),param_grid, scoring =  'f1');
    clf.fit(features_train, labels_train)
    predictions = clf.predict(features_test)
    print "Random Forest Parameters: ", clf.best_estimator_
        
    metric_scoring_RF[0] += clf.score(features_test, labels_test)
    metric_scoring_RF[1] += precision_score(labels_test, predictions)
    metric_scoring_RF[2] += recall_score(labels_test, predictions)
    metric_scoring_RF[3] += f1_score(labels_test, predictions)
    
    ### 7. Logistic Regression
    param_grid = {
    'penalty': ['l1','l2'],
    'C':[1e-2, 1e-1, 1, 10, 100]
    }
    clf = GridSearchCV(LogisticRegression(), param_grid, scoring = 'f1')
    clf.fit(features_train, labels_train)
    predictions = clf.predict(features_test)
    print "Logistic Regression Parameters: ", clf.best_estimator_
        
    metric_scoring_LR[0] += clf.score(features_test, labels_test)
    metric_scoring_LR[1] += precision_score(labels_test, predictions)
    metric_scoring_LR[2] += recall_score(labels_test, predictions)
    metric_scoring_LR[3] += f1_score(labels_test, predictions)

print ""
print "===== K-fold Validation: Comparing the different classifiers ====="
print "Naive Bayes:    ", np.array(metric_scoring_NB)/3.0
print "SVM:            ", np.array(metric_scoring_SVM)/3.0
print "Decision Trees: ", np.array(metric_scoring_DT)/3.0
print "kNN:            ", np.array(metric_scoring_kNN)/3.0
print "AdaBoost:       ", np.array(metric_scoring_AB)/3.0
print "Random Forest:  ", np.array(metric_scoring_RF)/3.0
print "Log Regression: ", np.array(metric_scoring_LR)/3.0


print ""
print "===== Doing StratifiedShuffleSplit via tester.py ====="

print "Naive Bayes:"
pipeline = Pipeline([
    ('scaler',StandardScaler()),
    ('clf', GaussianNB())
])
test_classifier(pipeline, my_dataset, features_list)

print ""
print "Support Vector Machines:"
pipeline = Pipeline([
    ('scaler',StandardScaler()),
    ('clf', SVC(C= 0.1, gamma= 0.1))
])
test_classifier(pipeline, my_dataset, features_list)

print ""
print "Decision Tree: "
pipeline = Pipeline([
    ('scaler',StandardScaler()),
    ('clf', DecisionTreeClassifier(min_samples_split=4))
])
test_classifier(pipeline, my_dataset, features_list)

print ""
print "k Nearest Neighbors: "
pipeline = Pipeline([
    ('scaler',StandardScaler()),
    ('clf', KNeighborsClassifier(n_neighbors=2, weights='distance'))
])
test_classifier(pipeline, my_dataset, features_list)

print ""
print "AdaBoost: "
pipeline = Pipeline([
    ('scaler',StandardScaler()),
    ('clf', AdaBoostClassifier(n_estimators=5, learning_rate=1))
])
test_classifier(pipeline, my_dataset, features_list)


print ""
print "Random Forest: "
pipeline = Pipeline([
    ('scaler',StandardScaler()),
    ('clf', RandomForestClassifier(n_estimators= 2, min_samples_split=4))
])
test_classifier(pipeline, my_dataset, features_list)

print ""
print "Logistic Regression: "
pipeline = Pipeline([
    ('scaler',StandardScaler()),
    ('clf', LogisticRegression(C=1, penalty='l1'))
])
test_classifier(pipeline, my_dataset, features_list)


print "Task 5: Tune your classfier"
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

print ""
print "===== Evaluation of Final Decision Tree  ===== "
pipeline = Pipeline([
    ('scaler',StandardScaler()),
    ('clf', DecisionTreeClassifier(min_samples_split=26))
])
test_classifier(pipeline, my_dataset, features_list)

pipeline.fit(features_scaled, labels)
print ""
print "===== Features Importances of Final Decision Tree  ===== "
print features_list
print pipeline.named_steps['clf'].feature_importances_

### "Task 6: Dump your classifier"
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(pipeline, my_dataset, features_list)