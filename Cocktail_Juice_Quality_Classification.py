#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import scale, StandardScaler
from sklearn.preprocessing import StandardScaler, Normalizer,MinMaxScaler 
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
import warnings
warnings.simplefilter(action='ignore')
sns.set()
plt.style.use("ggplot")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:





# In[2]:


Juice= pd.read_csv("Cocktail Juice Quality_Training Dataset.csv")


# In[3]:


Juice.head()


# In[4]:


Juice.info()


# In[5]:


Juice.describe().T


# In[6]:


Juice.columns


# In[7]:


print(Juice.isnull().sum())


# In[8]:


p25 = np.percentile(Juice.pH,25)
p75 = np.percentile(Juice.pH,75)
iqr = p75 - p25
cutoff = iqr*1.5
lower = p25-cutoff
upper = p75 + cutoff

outliers = [x for x in Juice.pH if x < lower or x > upper]
print('Lowest 5 outliers : ',sorted(outliers)[:5]) 
print('Highest 5 outliers : ',sorted(outliers)[-5:])


# In[9]:


plt.figure(figsize=(32,22))
plt.suptitle('Boxplots of each feature showing outliers',fontsize=24)
for i in range(1,Juice.shape[1]+1):
    plt.subplot(2,6,i)
    plt.boxplot(Juice.iloc[:,i-1])
    plt.title(Juice.columns[i-1],fontsize=18)


# In[10]:


Juice['quality'].unique()


# In[11]:


Juice.head()


# In[12]:


pd.DataFrame(Juice['quality'].value_counts())


# In[13]:


sns.countplot(Juice['quality'])
plt.show()


# In[14]:


print('Average Wine Quality =', Juice['quality'].mean())


# In[15]:


plt.figure(figsize=(10,7), dpi=100)
sns.heatmap(Juice.corr(), annot=True)


# In[16]:


bins = (2,6, 8)
labels = ['bad', 'good']
Juice['quality'] = pd.cut(x = Juice['quality'], bins = bins, labels = labels)
Juice['quality'].value_counts()


# In[17]:


sns.countplot(Juice['quality'])
plt.show()


# In[18]:


Juice.head()


# In[19]:


from sklearn.preprocessing import StandardScaler, LabelEncoder
label_quality = LabelEncoder()


# In[20]:


Juice['quality'] = label_quality.fit_transform(Juice['quality'])


# In[21]:


Juice.head()


# In[22]:


# wine['quality'].value_counts()


# In[23]:


x = Juice.drop(['quality'], axis=1)


# In[24]:


y = Juice['quality']


# In[25]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2 , random_state=24)


# In[26]:


sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)


# In[27]:


from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)
log_reg_pred_y = log_reg.predict(x_test)
print(classification_report(y_test, log_reg_pred_y))

print('Confusion Matrix\n', confusion_matrix(y_test, log_reg_pred_y))

log_reg_acc = accuracy_score(y_test, log_reg_pred_y)
print('\nAccuracy Score =', log_reg_acc*100, '%')


# In[28]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
rfc_pred_y = rfc.predict(x_test)
print(classification_report(y_test, rfc_pred_y))

print('Confusion Matrix\n', confusion_matrix(y_test, rfc_pred_y))

rfc_acc = accuracy_score(y_test, rfc_pred_y)
print('\nAccuracy Score =', rfc_acc*100, '%')


# In[29]:


# from sklearn.ensemble import RandomForestClassifier

# rfc = RandomForestClassifier()
# rfc.fit(x_train, y_train)
# rfc_pred_y = rfc.predict(x_test)
# print(classification_report(y_test, rfc_pred_y))
# print('Confusion Matrix\n', confusion_matrix(y_test, rfc_pred_y))

# rfc_acc = accuracy_score(y_test, rfc_pred_y)
# print('\nAccuracy Score =', rfc_acc*100, '%')


# In[30]:


from sklearn.svm import SVC

svc = SVC()
svc.fit(x_train,y_train)
svc_pred_y = svc.predict(x_test)
print(classification_report(y_test, svc_pred_y))
print('Confusion Matrix\n', confusion_matrix(y_test, svc_pred_y))

svc_acc = accuracy_score(y_test, svc_pred_y)
print('\nAccuracy Score =', svc_acc*100, '%')


# In[31]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
knn_pred_y = knn.predict(x_test)
print(classification_report(y_test, knn_pred_y))
print('Confusion Matrix\n', confusion_matrix(y_test, knn_pred_y))

knn_acc = accuracy_score(y_test, knn_pred_y)
print('\nAccuracy Score =', knn_acc*100, '%')


# In[32]:


from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
dtc_pred_y = dtc.predict(x_test)
print(classification_report(y_test, dtc_pred_y))
print('Confusion Matrix\n', confusion_matrix(y_test, dtc_pred_y))

dtc_acc = accuracy_score(y_test, dtc_pred_y)
print('\nAccuracy Score =', dtc_acc*100, '%')


# In[33]:


from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(x_train, y_train)
gnb_pred_y = gnb.predict(x_test)
print(classification_report(y_test, gnb_pred_y))
print('Confusion Matrix\n', confusion_matrix(y_test, gnb_pred_y))

gnb_acc = accuracy_score(y_test, gnb_pred_y)
print('\nAccuracy Score =', gnb_acc*100, '%')


# In[34]:


from xgboost import XGBClassifier
xg = XGBClassifier()
xg.fit(x_train, y_train)
xg_pred_y = xg.predict(x_test)
print(classification_report(y_test, xg_pred_y))
print('Confusion Matrix\n', confusion_matrix(y_test, xg_pred_y))

xg_acc = accuracy_score(y_test, xg_pred_y)
print('\nAccuracy Score =', xg_acc*100, '%')


# In[35]:


log_reg_scores = cross_val_score(log_reg, x_train, y_train, cv=10)
print(log_reg_scores)
print("Mean Score =", log_reg_scores.mean().round(5)*100,'%')


# In[36]:


rfc_scores = cross_val_score(rfc, x_train, y_train, cv=10)
print(rfc_scores)
print('Mean Score =', rfc_scores.mean().round(5)*100,'%')


# In[37]:


svc_scores = cross_val_score(svc, x_train, y_train, cv=10)
print(svc_scores)
print('Mean Scores =', svc_scores.mean().round(5)*100, '%')


# In[38]:


gnb_scores = cross_val_score(gnb, x_train, y_train, cv=10)
print(gnb_scores)
print('Mean Score =', gnb_scores.mean().round(5)*100, '%')


# In[39]:


dtc_scores = cross_val_score(dtc, x_train, y_train, cv=10)
print(dtc_scores)
print('Mean Score =', dtc_scores.mean().round(5)*100, '%')


# In[40]:


knn_scores = cross_val_score(knn, x_train, y_train, cv=10)
print(knn_scores)
print('Mean Score =', knn_scores.mean().round(5)*100, '%')


# In[41]:


xg_scores = cross_val_score(xg, x_train, y_train, cv=10)
print(xg_scores)
print("Mean Score =", xg_scores.mean().round(5)*100,'%')


# In[42]:


result = pd.DataFrame([{'Algorithm':'Logistic Regression', 'Accuracy Score (%)':log_reg_acc*100,'Cross Validation Score (%)':log_reg_scores.mean().round(5)*100},
                       {'Algorithm':'Random Forest Classifier', 'Accuracy Score (%)':rfc_acc*100, 'Cross Validation Score (%)':rfc_scores.mean().round(5)*100},
                       {'Algorithm':'Support Vector Classifier', 'Accuracy Score (%)':svc_acc*100, 'Cross Validation Score (%)':svc_scores.mean().round(5)*100},
                       {'Algorithm':'Decision Tree Classifier', 'Accuracy Score (%)':dtc_acc*100, 'Cross Validation Score (%)':dtc_scores.mean().round(5)*100},
                       {'Algorithm':'Gaussian Naive Bayes', 'Accuracy Score (%)':gnb_acc*100, 'Cross Validation Score (%)':gnb_scores.mean().round(5)*100},
                       {'Algorithm':'K-Nearest Neighbor', 'Accuracy Score (%)':knn_acc*100, 'Cross Validation Score (%)':knn_scores.mean().round(5)*100},
                      {'Algorithm':'XGBClassifier', 'Accuracy Score (%)':xg_acc*100, 'Cross Validation Score (%)':xg_scores.mean().round(5)*100}])
result.set_index(['Algorithm']).sort_values(by = ['Accuracy Score (%)'], ascending = False)


# In[43]:


# result = pd.DataFrame([{'Algorithm':'Logistic Regression', 'Accuracy Score (%)':log_reg_acc*100},
#                        {'Algorithm':'Random Forest Classifier', 'Accuracy Score (%)':rfc_acc*100},
#                        {'Algorithm':'Support Vector Classifier', 'Accuracy Score (%)':svc_acc*100},
#                        {'Algorithm':'Decision Tree Classifier', 'Accuracy Score (%)':dtc_acc*100},
#                        {'Algorithm':'Gaussian Naive Bayes', 'Accuracy Score (%)':gnb_acc*100},
#                        {'Algorithm':'K-Nearest Neighbor', 'Accuracy Score (%)':knn_acc*100},
#                       {'Algorithm':'XGBClassifier', 'Accuracy Score (%)':xg_acc*100}])
# result.set_index(['Algorithm']).sort_values(by = ['Accuracy Score (%)'], ascending = False)


# In[44]:


import matplotlib.pyplot as plt

# Accuracy scores for each model
models = ['Logistic Regression', 'Naive Bayes', 'K-Nearest Neighbors', 'Decision Tree', 'Support Vector Machines', 'Random Forest', 'XG Boost']
accuracy = [log_reg_acc, gnb_acc, knn_acc,dtc_acc,svc_acc,rfc_acc,xg_acc ]

# Multiply accuracy by 100
accuracy_percent = [acc * 100 for acc in accuracy]

# Plotting the accuracy scores
plt.figure(figsize=(10, 6))
bars = plt.bar(models, accuracy, color='skyblue')
plt.xlabel('Machine Learning Models')
plt.ylabel('Accuracy')
plt.title('Accuracy of Machine Learning Models')
plt.ylim(0.55, 1)  # Limiting y-axis from 0.7 to 1 for better visualization
plt.xticks(rotation=45)  # Rotating x-axis labels for better readability

# Adding accuracy values on top of each bar
for bar, acc in zip(bars, accuracy_percent):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.05, f'{acc:.2f}%', ha='center', color='black', fontsize=10)

plt.tight_layout()
plt.show()


# In[45]:


import matplotlib.pyplot as plt

# Accuracy scores for each model
models = [  'Support Vector Machines', 'Random Forest', 'XG Boost']
accuracy =[svc_acc,rfc_acc,xg_acc ]

# Multiply accuracy by 100
accuracy_percent = [acc * 100 for acc in accuracy]

# Plotting the accuracy scores
plt.figure(figsize=(10, 6))
bars = plt.bar(models, accuracy, color='skyblue')
plt.xlabel('Machine Learning Models')
plt.ylabel('Accuracy')
plt.title('Accuracy of Machine Learning Models')
plt.ylim(0.55, 1)  # Limiting y-axis from 0.7 to 1 for better visualization
plt.xticks(rotation=45)  # Rotating x-axis labels for better readability

# Adding accuracy values on top of each bar
for bar, acc in zip(bars, accuracy_percent):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.05, f'{acc:.2f}%', ha='center', color='black', fontsize=10)

plt.tight_layout()
plt.show()


# In[46]:


# models = ['SVC', ' Logistic Regression', 'Random Forest ','KNeighborsClassifier']
# accuracies = [svc_acc, log_reg_acc,rfc_acc,knn_acc]

# plt.bar(models, accuracies)
# plt.xlabel('Models')
# plt.ylabel('Accuracy')
# plt.title('Model Comparison')
# plt.ylim(0.2, 1)  # Adjust ylim for better visualization
# plt.show()


# In[47]:


# model = RandomForestClassifier()
# model.fit(x_train, y_train)
# model.score(x_test, y_test)


# In[48]:


# nine_inputs = []
# for i in range(9):
#     user_input = float(input(f"Enter input {i+1}: "))
#     nine_inputs.append(user_input)

# # Convert the list of inputs into a NumPy array
# features = np.array([nine_inputs])

# # Use the model to predict based on the input features
# prediction = model.predict(features)

# print("Prediction:", prediction)


# In[49]:


# from sklearn.ensemble import RandomForestClassifier

# # Assuming you have already split your data into training and testing sets as x_train, x_test, y_train, y_test

# # Initialize the RandomForestClassifier model
# model = RandomForestClassifier()

# # Train the model
# model.fit(x_train, y_train)

# # Evaluate the accuracy on the test set
# accuracy = model.score(x_test, y_test)
# print("Accuracy of the RandomForestClassifier model:", accuracy*100)


# In[50]:



from sklearn.svm import SVC

model = SVC()

# Train the model
model.fit(x_train, y_train)

# Evaluate the accuracy on the test set
accuracy = model.score(x_test, y_test)
print("Accuracy of the  model XG Boost:", accuracy*100)


# In[53]:



nine_inputs = []
for i in range(9):
    user_input = float(input(f"Enter input {i+1}: "))
    nine_inputs.append(user_input)

# Convert the list of inputs into a NumPy array
features = np.array([nine_inputs])

# Use the model to predict based on the input features
prediction = model.predict(features)

print("Prediction:", prediction)


# In[54]:


from xgboost import XGBClassifier

# Assuming you have already split your data into training and testing sets as x_train, x_test, y_train, y_test

# Initialize the Logistic Regression model
model = XGBClassifier()

# Train the model
model.fit(x_train, y_train)

# Evaluate the accuracy on the test set
accuracy = model.score(x_test, y_test)
print("Accuracy of the  model XG Boost:", accuracy*100)


# In[55]:


nine_inputs = []
for i in range(9):
    user_input = float(input(f"Enter input {i+1}: "))
    nine_inputs.append(user_input)

# Convert the list of inputs into a NumPy array
features = np.array([nine_inputs])

# Use the model to predict based on the input features
prediction = model.predict(features)

print("Prediction:", prediction)


# In[56]:


from sklearn.ensemble import RandomForestClassifier

# Assuming you have already split your data into training and testing sets as x_train, x_test, y_train, y_test

# Initialize the RandomForestClassifier model
model = RandomForestClassifier()

# Train the model
model.fit(x_train, y_train)

# Evaluate the accuracy on the test set
accuracy = model.score(x_test, y_test)
print("Accuracy of the RandomForestClassifier model:", accuracy*100)


# In[57]:


nine_inputs = []
for i in range(9):
    user_input = float(input(f"Enter input {i+1}: "))
    nine_inputs.append(user_input)

# Convert the list of inputs into a NumPy array
features = np.array([nine_inputs])

# Use the model to predict based on the input features
prediction = model.predict(features)

print("Prediction:", prediction)


# In[86]:


features = np.array([[9.5, 7, 0,1.9,0.076, 25, 0.99,3.5,0.5]])
model.predict(features)


# In[ ]:




