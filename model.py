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


# indepedent and dependent columns
attrObese = ["Gender", "Age", "Height", "Weight", "Family", "FAVC", "Smoke", "CH2O", "FAF", "CALC"]
x = data[attrObese]
y = data['NObeyesdad']

# split in train and test
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=2)

sc = StandardScaler();
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# classifier = DecisionTreeClassifier()
# classifier.fit(x_train, y_train)

# y_pred = classifier.predict(x_test)

# # #calculate Accuracy
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# # #save the model
# file = open("model.pkl", 'wb')
# pickle.dump(classifier, file)


# names = ["KNearest_Neighbors", "SVM", "Decision_Tree"]

# classifiers = [
#     KNeighborsClassifier(3),
#     svm.SVC(),
#     DecisionTreeClassifier()]

# scores = []
# for name, clf in zip(names, classifiers):
#     clf.fit(x_train, y_train)
#     y_pred = clf.predict(x_test)
#     score = metrics.accuracy_score(y_test, y_pred)
#     scores.append(score)     

# df = pd.DataFrame()
# df['name'] = names
# df['score'] = scores
# print(df)

# print(score.max())


# prepare configuration for cross validation test harness
seed = 7

# prepare models
models = []

models.append(('KNN', KNeighborsClassifier()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('SVM', svm.SVC()))

# evaluate each model in turn
results = []
names = []
acc_mean = []
scoring = 'accuracy'
for name, model in models:
	
	kfold = model_selection.KFold(n_splits=10,shuffle=True, random_state=seed)
	cv_results = model_selection.cross_val_score(model, x, y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	acc_mean.append(cv_results.mean())
	print(acc_mean)
	high_acc = acc_mean.index(np.max(acc_mean))
	print(high_acc)
	classify = models[high_acc]
	print(classify)
	final = classify[1]
	print(final)

	# msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	# print(msg)

classifier = final
classifier.fit(x_train, y_train)

# y_pred = classifier.predict(x_test)

# 	# #calculate Accuracy
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# # #save the model
file = open("model.pkl", 'wb')
pickle.dump(classifier, file)
    



# 	model.fit(x_train, y_train)
# 	prediction = model.predict(x_test)
# 	print(name)
# 	print(metrics.accuracy_score(y_test,prediction))
# 	# print(metrics.classification_report(y_test,prediction))

	
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()