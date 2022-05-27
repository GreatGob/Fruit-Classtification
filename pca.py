import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

from sklearn import preprocessing
from sklearn import utils

# importing or loading the dataset
dataset = pd.read_excel(r'C:\Users\USER\OneDrive\Thematic\ex4\ex4\Hu_moment\hu_moment_berry.xlsx')
 
# distributing the dataset into two components X and Y
X = dataset.iloc[:, 0:7].values
Y = dataset.iloc[:, 6].values

# Splitting the X and Y into the training set and Testing set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# performing preprocessing part
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying PCA function on training and testing set of X component
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

explained_variance = pca.explained_variance_ratio_

#convert Y values to categorical values
lab = preprocessing.LabelEncoder()
Y_train_transformed = lab.fit_transform(Y_train)
Y_test_transformed = lab.fit_transform(Y_test)

# Fitting Logistic Regression To the training set
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train_transformed)

# Predicting the test set result using
# predict function under LogisticRegression
Y_pred = classifier.predict(X_test)

# making confusion matrix between test set of Y and predicted value.
cm = confusion_matrix(Y_test_transformed, Y_pred)



#Train
# Predicting the training set result through scatter plot
X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,
					stop = X_set[:, 0].max() + 1, step = 0.01),
					np.arange(start = X_set[:, 1].min() - 1,
					stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),
			X2.ravel()]).T).reshape(X1.shape), alpha = 0.75,
			cmap = ListedColormap(('yellow', 'white', 'aquamarine')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(Y_set)):
	plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
				c = ListedColormap(('red', 'green', 'blue'))(i), label = j)

plt.title('Logistic Regression (Training set)')
plt.xlabel('PC1') # for Xlabel
plt.ylabel('PC2') # for Ylabel
plt.legend() # to show legend

# show scatter plot
plt.show()



#Test
# Visualising the Test set results through scatter plot
X_set, Y_set = X_test, Y_test
 
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,
                     stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1,
                     stop = X_set[:, 1].max() + 1, step = 0.01))
 
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),
             X2.ravel()]).T).reshape(X1.shape), alpha = 0.75,
             cmap = ListedColormap(('yellow', 'white', 'aquamarine')))
 
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
 
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
 
# title for scatter plot
plt.title('Logistic Regression (Test set)')
plt.xlabel('PC1') # for Xlabel
plt.ylabel('PC2') # for Ylabel
plt.legend()

# show scatter plot
plt.show() 


#export to excel
#print("______")
#print(Y_set)

#data_train= pd.DataFrame(Y_set)
#data_train.to_excel('pca_train_berry.xlsx', sheet_name='Sheet 1', index = False)
