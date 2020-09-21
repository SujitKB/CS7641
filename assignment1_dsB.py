'''
GT User ID: sbiswas67	         Assignment 1 (Supervised Learning)
Dataset B:  Statlog (Vehicle Silhouettes) Dataset
'''
import pandas as pd
import numpy as np
import time as t
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import cross_validate

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

from preprocess_data import vehicle_silhouettes

X, Y = vehicle_silhouettes ("Vehicle_Silhouettes.csv")

# Split data into Train and Test datasets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.3, random_state=42)
train_sizes = np.linspace(0.01, 1.0, 5)

# Classifier Comparison metrics
class_accuracy   = [0.] * 5
class_train_time = [0.] * 5
class_query_time = [0.] * 5

## ################################################################################################################## ##
## ########################################### 1. Decision Tree ##################################################### ##
## ################################################################################################################## ##

############ Learning Curve - Default Hyperparameters ####################
train_sizes, train_scores, validation_scores = learning_curve(DecisionTreeClassifier(random_state=42),
                                                              X_train, Y_train,
                                                              train_sizes=train_sizes, cv=5)

clf_dt = DecisionTreeClassifier(random_state=42)
clf_dt.fit(X_train, Y_train)
Y_pred = clf_dt.predict(X_test)
dt_accuracy = clf_dt.score(X_test, Y_test) * 100
print('Accuracy of decision tree without hyperparameter tuning is %.2f%%' % (dt_accuracy))

plt.figure()
plt.plot(train_sizes, train_scores.mean(axis=1),marker='o', label='Training score')
plt.plot(train_sizes, validation_scores.mean(axis=1),marker='o', label='Cross-validation score')
plt.title('Learning Curve - Decision Tree Classifier')
plt.xlabel('No. of Training Instances')
plt.ylabel("Classification Score")
plt.legend()
plt.grid()
plt.savefig('DsB_DT_LC_NoTuning')
#plt.show()
plt.clf()


### Validation Curve - max_depth
depth_range = np.arange(0,20) + 1
train_scores, validation_scores = validation_curve(DecisionTreeClassifier(random_state=42),
                                                   X_train, Y_train,
                                                   param_name="max_depth",
                                                   param_range=depth_range, cv=5)

plt.plot(depth_range, train_scores.mean(axis=1), marker='o', label='Training score')
plt.plot(depth_range, validation_scores.mean(axis=1), marker='o', label='Cross-validation score')
plt.title('Validation Curve - Decision Tree Classifier')
plt.xlabel('max_depth')
plt.ylabel("Classification Score")
plt.legend()
plt.grid()
plt.savefig('DsB_DT_VC_max_depth')
#plt.show()
plt.clf()

### Validation Curve - min_samples_leaf
min_leaf_range = np.arange(0, 10) + 1
train_scores, validation_scores = validation_curve(DecisionTreeClassifier(random_state=42, max_depth=12),
                                                   X_train, Y_train,
                                                   param_name="min_samples_leaf", param_range=min_leaf_range,
                                                   cv=5)

plt.plot(min_leaf_range, train_scores.mean(axis=1),marker='o', label='Training score')
plt.plot(min_leaf_range, validation_scores.mean(axis=1),marker='o', label='Cross-validation score')
plt.title('Validation Curve - Decision Tree Classifier')
plt.xlabel('min_samples_leaf')
plt.ylabel("Classification Score")
plt.legend()
plt.grid()
plt.savefig('DsB_DT_VC_min_samples_leaf')
#plt.show()
plt.clf()

### Validation Curve - min_samples_split
min_split_range = np.arange(1, 20) + 1
train_scores, validation_scores = validation_curve(DecisionTreeClassifier(random_state=42,
                                                                          max_depth=12, min_samples_leaf=2), X_train, Y_train, param_name="min_samples_split", param_range=min_split_range, cv=5)

plt.plot(min_split_range, train_scores.mean(axis=1),marker='o', label='Training score')
plt.plot(min_split_range, validation_scores.mean(axis=1),marker='o', label='Cross-validation score')
plt.title('Validation Curve - Decision Tree Classifier')
plt.xlabel('min_samples_split')
plt.ylabel("Classification Score")
plt.legend()
plt.grid()
plt.savefig('DsB_DT_VC_min_samples_split.png')
#plt.show()
plt.clf()

### Validation Curve - max_features
feature_range = np.arange(0, np.shape(X_train)[1]) + 1
train_scores, validation_scores = validation_curve(DecisionTreeClassifier(random_state=42, max_depth=12, min_samples_leaf=2, min_samples_split=18), X_train, Y_train, param_name="max_features", param_range=feature_range, cv=5)

plt.plot(feature_range, train_scores.mean(axis=1),marker='o', label='Training score')
plt.plot(feature_range, validation_scores.mean(axis=1),marker='o', label='Cross-validation score')
plt.title('Validation Curve - Decision Tree Classifier')
plt.xlabel('max_features')
plt.ylabel("Classification Score")
plt.legend()
plt.grid()
plt.savefig('DsB_DT_VC_max_features.png')
#plt.show()
plt.clf()


############ Learning Curve - Tuned Hyperparameters ####################
train_sizes = np.linspace(0.01, 1.0, 5)
train_sizes, train_scores, validation_scores = learning_curve(DecisionTreeClassifier(random_state=42,max_depth=12,min_samples_leaf=2,min_samples_split=18,max_features=18),
                                                              X_train, Y_train, train_sizes=train_sizes, cv=5)
plt.figure()
#plt.style.use('seaborn')
plt.plot(train_sizes, train_scores.mean(axis=1),marker='o', label='Training score')
plt.plot(train_sizes, validation_scores.mean(axis=1),marker='o', label='Cross-validation score')
plt.title('Learning Curve - Tuned Decision Tree Classifier')
plt.xlabel('No. of Training Instances')
plt.ylabel("Classification Score")
plt.legend()
plt.grid()
plt.savefig('DsB_DT_LC_Tuned.png')
#plt.show()
plt.clf()

clf_dt = DecisionTreeClassifier(random_state=42,max_depth=12,min_samples_leaf=2,min_samples_split=18,max_features=18)

tStart = t.time()
clf_dt.fit(X_train, Y_train)
tEnd = t.time()
class_train_time[0] = tEnd - tStart

tStart = t.time()
Y_pred = clf_dt.predict(X_test)
tEnd = t.time()
class_query_time[0] = tEnd - tStart

dt_accuracy = clf_dt.score(X_test, Y_test) * 100
class_accuracy[0] = dt_accuracy
print('Accuracy of decision tree WITH hyperparameter tuning is %.2f%%' % (dt_accuracy))

## ################################################################################################################## ##
## ############################################### 2. Boosting (AdaBoost) ########################################### ##
## ################################################################################################################## ##

'''for depth in [1,5,8]:
    for learning_rate in [1, 1.5]:
        for n_estimators in np.arange(1, 1050, 50):
            clf_dt = DecisionTreeClassifier(random_state=42, max_depth=depth)

            clf_adab = AdaBoostClassifier(clf_dt, learning_rate=learning_rate, n_estimators=n_estimators, algorithm='SAMME', random_state=42)
            clf_adab.fit(X_train, Y_train)
            Y_pred = clf_adab.predict(X_test)
            clf_adab_accuracy = clf_adab.score(X_test, Y_test) * 100


            print('Accuracy of AdaBoost with depth = %d , n_estimators = %d and learning_rate = %.2f is %.2f%%' % (depth, n_estimators, learning_rate, clf_adab_accuracy))
'''

############ Learning Curve - UnTuned Hyperparameters ####################
train_sizes = np.linspace(0.01, 1.0, 5)
clf_dt = DecisionTreeClassifier(random_state=42, max_depth=1)
clf_adab = AdaBoostClassifier(clf_dt, learning_rate=1, n_estimators=151, random_state=42)
train_sizes, train_scores, validation_scores = learning_curve(clf_adab,
                                                              X_train, Y_train, train_sizes=train_sizes, cv=5)
plt.figure()
plt.plot(train_sizes, train_scores.mean(axis=1),marker='o', label='Training score')
plt.plot(train_sizes, validation_scores.mean(axis=1),marker='o', label='Cross-validation score')
plt.title('Learning Curve - Base AdaBoost Classifier')
plt.xlabel('No. of Training Instances')
plt.ylabel("Classification Score")
plt.legend()
plt.grid()
plt.savefig('DsB_AB_LC_Tuned_1_151_1.png')
#plt.show()
plt.clf()

## Validation curve
n_estimators_range = np.arange (50, 1050, 50)
clf_dt = DecisionTreeClassifier(random_state=42, max_depth=1)
train_scores, validation_scores = validation_curve(AdaBoostClassifier(clf_dt, learning_rate=0.5, random_state=42),
                                                   X_train, Y_train, param_name="n_estimators", param_range=n_estimators_range, cv=5)

plt.plot(n_estimators_range, train_scores.mean(axis=1),marker='o', label='Training score')
plt.plot(n_estimators_range, validation_scores.mean(axis=1),marker='o', label='Cross-validation score')
plt.title('Validation Curve - AdaBoost (max_depth=1, learning_rate=0.05')
plt.xlabel('n_estimators')
plt.ylabel("Classification Score")
plt.legend()
plt.grid()
plt.savefig('DsB_AB_VC_n_estimators_1_0.5.png')
plt.clf()

# Accuracy on Test dataset with Optimal Hyper-parameters
adab = AdaBoostClassifier(DecisionTreeClassifier(random_state=42, max_depth=5),
                                learning_rate=1.5, n_estimators=101, random_state=42)

tStart = t.time()
adab.fit(X_train, Y_train)
tEnd = t.time()
class_train_time[1] = tEnd - tStart

tStart = t.time()
Y_pred = adab.predict(X_test)
tEnd = t.time()
class_query_time[1] = tEnd - tStart

adab_accuracy = adab.score(X_test, Y_test) * 100
class_accuracy[1] = adab_accuracy
print('Accuracy of AdaBoost is %.2f%%' % (adab_accuracy))

## ################################################################################################################## ##
## ############################################ 3. Artificial NN #################################################### ##
## ################################################################################################################## ##

mlp = MLPClassifier(hidden_layer_sizes=(4, 3), random_state=42, max_iter=4000)
mlp.fit(X_train, Y_train)
Y_pred = mlp.predict(X_test)
accuracy_mlp = mlp.score(X_test, Y_test) * 100
print('Accuracy of ANN without any tuning is %.2f%%' % (accuracy_mlp))

train_sizes, train_scores, validation_scores = learning_curve(mlp,
                                                              X_train, Y_train,
                                                              train_sizes=train_sizes, cv=5)

plt.figure()
plt.plot(train_sizes, train_scores.mean(axis=1),marker='o', label='Training score')
plt.plot(train_sizes, validation_scores.mean(axis=1),marker='o', label='Cross-validation score')
plt.title('Learning Curve - ANN Classifier')
plt.xlabel('No. of Training Instances')
plt.ylabel("Classification Score")
plt.legend()
plt.grid()
plt.savefig('DsB_ANN_LC_NoTuning.png')
plt.clf()

# Alpha (Regularization parameter)
alpha_range = np.logspace(-10, 3, 7)
train_scores, test_scores = validation_curve(mlp, X_train, Y_train, param_name="alpha", param_range=alpha_range, cv=5)

plt.figure()
plt.semilogx(alpha_range, np.mean(train_scores, axis=1), label='Training score')
plt.semilogx(alpha_range, np.mean(test_scores, axis=1), label='Cross-validation score')
plt.title('Model Complexity Curve - ANN')
plt.xlabel('Alpha')
plt.ylabel("Classification Score")
plt.grid()
plt.savefig('DsB_ANN_VC_Alpha.png')

# Learning_Rate
learningR_range = np.logspace(-5, 0, 6)
train_scores, test_scores = validation_curve(mlp, X_train, Y_train,
                                             param_name="learning_rate_init", param_range=learningR_range,
                                             cv=5)
plt.figure()
plt.semilogx(learningR_range, np.mean(train_scores, axis=1), label='Training score')
plt.semilogx(learningR_range, np.mean(test_scores, axis=1), label='Cross-validation score')
plt.title('Model Complexity Curve - ANN')
plt.xlabel('Learning Rate')
plt.ylabel("Classification Score")
plt.grid()
plt.savefig('DsB_ANN_VC_LearningRate.png')

# ANN Classifier with tuned Hyperparameter
mlp = MLPClassifier(hidden_layer_sizes=(4, 3), alpha=0.01, learning_rate_init=0.00398, random_state=42, max_iter=4000)

tStart = t.time()
mlp.fit(X_train, Y_train)
tEnd = t.time()
class_train_time[2] = tEnd - tStart

tStart = t.time()
Y_pred = mlp.predict(X_test)
tEnd = t.time()
class_query_time[2] = tEnd - tStart

accuracy_mlp = mlp.score(X_test, Y_test) * 100
class_accuracy[2] = accuracy_mlp
print('Accuracy of ANN with tuning is %.2f%%' % (accuracy_mlp))

train_sizes, train_scores, validation_scores = learning_curve(mlp,
                                                              X_train, Y_train,
                                                              train_sizes=train_sizes, cv=5)

plt.figure()
plt.plot(train_sizes, train_scores.mean(axis=1),marker='o', label='Training score')
plt.plot(train_sizes, validation_scores.mean(axis=1),marker='o', label='Cross-validation score')
plt.title('Learning Curve - ANN Classifier')
plt.xlabel('No. of Training Instances')
plt.ylabel("Classification Score")
plt.legend()
plt.grid()
plt.savefig('DsB_ANN_LC_Tuned.png')
plt.clf()

## ################################################################################################################## ##
## ##################################### 4. Support Vector Machine (Linear/rbf) ##################################### ##
## ################################################################################################################## ##

#### kernel='rbf'
#Initializing a SVM model (untuned)
svm_rbf = SVC(random_state=42, kernel='rbf')

#Fitting the model to the training data
svm_rbf.fit(X_train, Y_train)

#Extracting the accuracy score from the training data
accuracy_svm = svm_rbf.score(X_test, Y_test) * 100
print('Accuracy of SVM (RBF) without any tuning is %.2f%%' % (accuracy_svm))

# C (Inverse Regularization parameter)
c_range = np.logspace(-4, 5, 5)
train_scores, test_scores = validation_curve(svm_rbf, X_train, Y_train, param_name="C", param_range=c_range,
                                             cv=5)

plt.figure()
plt.semilogx(c_range, np.mean(train_scores, axis=1), label='Training score')
plt.semilogx(c_range, np.mean(test_scores, axis=1), label='Cross-validation score')
plt.title('Model Complexity Curve - SVM (RBF)')
plt.xlabel('C (Inverse Regularization parameter)')
plt.ylabel("Classification Score")
plt.grid()
plt.savefig('DsB_SVM_rbf_VC.png')

# SVM-RBF Classifier with tuned Hyper-parameter
svm_rbf = SVC(random_state=42, kernel='rbf',C=10)

tStart = t.time()
svm_rbf.fit(X_train, Y_train)
tEnd = t.time()
class_train_time[3] = tEnd - tStart

tStart = t.time()
Y_pred = svm_rbf.predict(X_test)
tEnd = t.time()
class_query_time[3] = tEnd - tStart

accuracy_svm = svm_rbf.score(X_test, Y_test) * 100
class_accuracy[3] = accuracy_svm
print('Accuracy of SVM(RBF) with tuning is %.2f%%' % (accuracy_svm))

train_sizes, train_scores, validation_scores = learning_curve(svm_rbf,
                                                              X_train, Y_train,
                                                              train_sizes=train_sizes, cv=5)

plt.figure()
plt.plot(train_sizes, train_scores.mean(axis=1),marker='o', label='Training score')
plt.plot(train_sizes, validation_scores.mean(axis=1),marker='o', label='Cross-validation score')
plt.title('Learning Curve - SVM (RBF) Classifier')
plt.xlabel('No. of Training Instances')
plt.ylabel("Classification Score")
plt.legend()
plt.grid()
plt.savefig('DsB_SVM-RBF_LC_Tuned.png')
plt.clf()

##### kernel='linear'
#Initializing a SVM model (untuned)
svm_linear = SVC(random_state=42, kernel='linear')

#Fitting the model to the training data
svm_linear.fit(X_train, Y_train)

#Extracting the accuracy score from the training data
accuracy_svm = svm_linear.score(X_test, Y_test) * 100
print('Accuracy of SVM (Linear) without any tuning is %.2f%%' % (accuracy_svm))

# C (Inverse Regularization parameter)
c_range = np.logspace(-4, 5, 5)
train_scores, test_scores = validation_curve(svm_linear, X_train, Y_train, param_name="C", param_range=c_range,
                                             cv=5)

plt.figure()
plt.semilogx(c_range, np.mean(train_scores, axis=1), label='Training score')
plt.semilogx(c_range, np.mean(test_scores, axis=1), label='Cross-validation score')
plt.title('Model Complexity Curve - SVM (Linear)')
plt.xlabel('C (Inverse Regularization parameter)')
plt.ylabel("Classification Score")
plt.grid()
plt.savefig('DsB_SVM_linear_VC.png')

# SVM-Linear Classifier with tuned Hyper-parameter
svm_linear = SVC(random_state=42, kernel='linear', C=7)
svm_linear.fit(X_train, Y_train)
accuracy_svm = svm_linear.score(X_test, Y_test) * 100
print('Accuracy of SVM (Linear) with tuning is %.2f%%' % (accuracy_svm))

train_sizes, train_scores, validation_scores = learning_curve(svm_linear,
                                                              X_train, Y_train,
                                                              train_sizes=train_sizes, cv=5)

plt.figure()
plt.plot(train_sizes, train_scores.mean(axis=1),marker='o', label='Training score')
plt.plot(train_sizes, validation_scores.mean(axis=1),marker='o', label='Cross-validation score')
plt.title('Learning Curve - SVM (Linear) Classifier')
plt.xlabel('No. of Training Instances')
plt.ylabel("Classification Score")
plt.legend()
plt.grid()
plt.savefig('DsB_SVM-Linear_LC_Tuned.png')
plt.clf()

## ################################################################################################################## ##
## ############################################# 4. K Nearest Neighbour (KNN) ####################################### ##
## ################################################################################################################## ##


# n_neighbours (Number of nearest neighbours)
k_range = np.arange(1, 200) + 5
train_scores, validation_scores = validation_curve(KNeighborsClassifier (), X_train, Y_train,
                                                    param_name="n_neighbors", param_range=k_range, cv=5)

plt.figure()
plt.plot(k_range, train_scores.mean(axis=1), label='Training score')
plt.plot(k_range, validation_scores.mean(axis=1), label='Cross-validation score')
plt.title('Model Complexity Curve - KNN')
plt.xlabel('Number of Nearest Neighbours')
plt.ylabel("Classification Score")
plt.grid()
plt.savefig('DsB_KNN_VC.png')

#Fitting the model to the training data
knn = KNeighborsClassifier (n_neighbors=5)  #Optimal

tStart = t.time()
knn.fit(X_train, Y_train)
tEnd = t.time()
class_train_time[4] = tEnd - tStart

tStart = t.time()
Y_pred = knn.predict(X_test)
tEnd = t.time()
class_query_time[4] = tEnd - tStart

#Extracting the accuracy score from the training data
accuracy_knn = knn.score(X_test, Y_test) * 100
class_accuracy[4] = accuracy_knn
print('Accuracy of KNN after tuning is %.2f%%' % (accuracy_knn))

train_sizes = np.linspace(0.1, 1.0, 5)

# Learning curve - Optimal K
train_sizes, train_scores, validation_scores = learning_curve(knn,
                                                              X_train, Y_train,
                                                              train_sizes=train_sizes, cv=5)

plt.figure()
plt.plot(train_sizes, train_scores.mean(axis=1),marker='o', label='Training score')
plt.plot(train_sizes, validation_scores.mean(axis=1),marker='o', label='Cross-validation score')
plt.title('Learning Curve - KNN Classifier (Optimal K')
plt.xlabel('No. of Training Instances')
plt.ylabel("Classification Score")
plt.legend()
plt.grid()
plt.savefig('DsB_KNN_LC_Tuned.png')
plt.clf()

################ Comparison ###################
learners = ('Decision tree', 'AdaBoost', 'Neural Network', 'SVM', 'kNN')
y_pos = np.arange(len(learners))

## Accuracy Comparison
plt.figure(figsize=(10,15))
plt.rcParams.update({'font.size':20})
plt.rcParams['axes.labelweight'] = 'bold'

plt.bar(y_pos, class_accuracy, width=0.5)
plt.xticks(y_pos, learners, rotation=30)
plt.grid(True)
plt.ylim((70, 90))
plt.title('Comparison of Maximum Accuracy Score')
plt.ylabel('Accuracy')
plt.savefig('DsB_Accuracy_Classifiers.png')
plt.clf()

# ####################Training Time
plt.figure(figsize=(10,15))
plt.rcParams.update({'font.size':20})
plt.rcParams['axes.labelweight'] = 'bold'

plt.bar(y_pos, class_train_time, width=0.5)
plt.xticks(y_pos, learners, rotation=30)
plt.grid(True)
plt.ylim((0,1))
plt.title('Comparison of Training Time')
plt.ylabel('Train Time (sec)')
plt.savefig('DsB_TrainTime_Classifiers.png')
plt.clf()

# ###################Query/Predict Time

plt.figure(figsize=(12,15))
plt.rcParams.update({'font.size':20})
plt.rcParams['axes.labelweight'] = 'bold'

plt.bar(y_pos, class_query_time, width=0.5)
plt.xticks(y_pos, learners, rotation=30)
plt.grid(True)
plt.ylim((0, 0.02))
plt.title('Comparison of Query Time')
plt.ylabel('Query Time (seconds)')
plt.savefig('DsB_QueryTime_Classifiers.png')
plt.clf()