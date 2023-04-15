
#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

df=pd.read_csv(r'C:/Users/krish/Machine Learning/diabetes.csv')

y=df['class']
x=df[['Gender_man','Polyuria','Polydipsia','sudden weight loss','weakness','Polyphagia','Genital thrush','visual blurring','Itching','Irritability','delayed healing','partial paresis', 'muscle stiffness','Alopecia','Obesity']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
gnb = GaussianNB()
gnb.fit(x_train, y_train)
predict=gnb.predict(x_test)
print(pd.DataFrame(confusion_matrix(y_test,predict),columns=['Predicted No', 'Predicted Yes'],index=['Actual No', 'Actual Yes']))
print(classification_report(y_test,predict))

cv = KFold(n_splits=10, random_state=1, shuffle=True)
scores = cross_val_score(gnb, x, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('Cross Fold Validation Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
# %%
