#%%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn import tree
from matplotlib import pyplot as plt
from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

df=pd.read_csv(r'C:/Users/krish/Machine Learning/diabetes.csv')

y=df['class']
x=df[['Gender_man','Polyuria','Polydipsia','sudden weight loss','weakness','Polyphagia','Genital thrush','visual blurring','Itching','Irritability','delayed healing','partial paresis','muscle stiffness','Alopecia','Obesity']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
clf = DecisionTreeClassifier()
clf=clf.fit(x_train, y_train)
predictions=clf.predict(x_test)

accuracy_score(y_test, predictions)
confusion_matrix(y_test, predictions, labels=[0,1])
precision_score(y_test, predictions)

print(pd.DataFrame(confusion_matrix(y_test,predictions),columns=['Predicted No', 'Predicted Yes'],index=['Actual No', 'Actual Yes']))
print(classification_report(y_test,predictions))
cv = KFold(n_splits=10, random_state=1, shuffle=True)
scores = cross_val_score(clf, x, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('Cross Fold Validation Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(clf,  
                   class_names={0:'Negative', 1:'Positive'},
                   filled=True,
                  fontsize=12)


