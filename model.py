import pandas as pd
import numpy as np
from sklearn import metrics, svm
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from six import StringIO  
from IPython.display import Image  
import pickle
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv(r"C:\Users\user\Downloads\ObesityData.csv")
print(data)
le = LabelEncoder()
le.fit(data['Gender'])
data['Gender'] = le.transform(data['Gender'])
le.fit(data['Family'])
data['Family'] = le.transform(data['Family'])
le.fit(data['FAVC'])
data['FAVC'] = le.transform(data['FAVC'])
le.fit(data['Smoke'])
data['Smoke'] = le.transform(data['Smoke'])
le.fit(data['CH2O'])
data['CH2O'] = le.transform(data['CH2O'])
le.fit(data['FAF'])
data['FAF'] = le.transform(data['FAF'])
le.fit(data['CALC'])
data['CALC'] = le.transform(data['CALC'])


#indepedent and dependent columns
attrObese= ["Gender", "Age", "Height", "Weight", "Family", "FAVC", "Smoke", "CH2O", "FAF", "CALC"]
x = data[attrObese]
y = data['NObeyesdad']
#split in train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

classifier = DecisionTreeClassifier()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

# #calculate Accuracy
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# #save the model
file = open("model.pkl", 'wb')
pickle.dump(classifier, file)