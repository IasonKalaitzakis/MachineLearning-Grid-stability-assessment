# evaluate RFE for classification
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn import metrics
from matplotlib import pyplot
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

# create dataset
data = pd.read_csv('smart_grid_stability_augmented.csv')
map1 = {'unstable': 0, 'stable': 1}
data['stabf'] = data['stabf'].replace(map1)
data = data.sample(frac=1)

X,z,y=np.hsplit(data, np.array([12,13]))

X.drop('p1', inplace=True, axis=1)
X.drop('p2', inplace=True, axis=1)
X.drop('p3', inplace=True, axis=1)
X.drop('p4', inplace=True, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=1)

clf = XGBClassifier(use_label_encoder=False)
clf.fit(X_train, y_train.values.ravel())
print("\nClassifier trained.")

#Predict the response for test dataset
y_pred = clf.predict(X_test)


lr_probs = clf.predict_proba(X_test)    # predict probabilities
lr_probs = lr_probs[:, 1]   # keep probabilities for the positive outcome only

# Model Accuracy: how often is the classifier correct?
print(f'Accuracy: {round(metrics.accuracy_score(y_test, y_pred),2)}')

cm = metrics.confusion_matrix(y_test, y_pred)


# Model Precision: What proportion of positive identifications was actually correct?
print(f'Precision: {round(metrics.precision_score(y_test, y_pred),2)}')

# Model Recall: What proportion of actual positives was identified correctly?
print(f'Recall: {round(metrics.recall_score(y_test, y_pred),2)}')

TP = cm[1][1]
TN = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]

sensitivity = (TP / float(TP + FN))
specificity = (TN / float(TN + FP))

print(f'Sensitivity: {round(sensitivity,2)}')
print(f'Specificity: {round(specificity,2)}\n')

cm_df = pd.DataFrame(cm, 
    columns = ['Predicted Negative', 'Predicted Positive'],
    index = ['Actual Negative', 'Actual Positive'])
print(cm_df)

# calculate AUC scores
lr_auc = roc_auc_score(y_test, lr_probs)
print('\nROC AUC=%.3f' % (lr_auc))

# calculate roc curves
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
# plot the roc curve for the model
pyplot.plot(lr_fpr, lr_tpr, marker='.')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
#show the plot
pyplot.show()
