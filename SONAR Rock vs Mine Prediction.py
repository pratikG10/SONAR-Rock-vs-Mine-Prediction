#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix


# In[2]:


sonar_data = pd.read_csv('sonar.csv', header=None)


# In[3]:


sonar_data.head()


# In[4]:


sonar_data.tail()


# In[5]:


sonar_data.shape


# In[6]:


sonar_data.groupby(60).mean()


# In[7]:


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'sonar_data' is your DataFrame and the target variable is in column 60
sns.countplot(x=sonar_data[60])
plt.title('Countplot of Target Variable')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()


# In[8]:


# Separating data and labels
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]


# In[9]:


# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.19, stratify=Y, random_state=2)


# In[10]:


# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[11]:


# Logistic Regression Model Training
logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(X_train, Y_train)


# In[12]:


# Evaluate Logistic Regression Model
training_data_accuracy_lr = accuracy_score(logistic_regression_model.predict(X_train), Y_train)
test_data_accuracy_lr = logistic_regression_model.score(X_test, Y_test)


# In[13]:


# Random Forest Model Training
random_forest_model = RandomForestClassifier(random_state=42)
random_forest_model.fit(X_train_scaled, Y_train)


# In[14]:


# Evaluate Random Forest Model
training_data_accuracy_rf = accuracy_score(random_forest_model.predict(X_train_scaled), Y_train)
test_data_accuracy_rf = accuracy_score(random_forest_model.predict(X_test_scaled), Y_test)




# In[15]:


print('\nLogistic Regression Model:')
print('Accuracy on training data: ', training_data_accuracy_lr)
print('Accuracy on test data: ', test_data_accuracy_lr)



# In[16]:


print('\nRandom Forest Model:')
print('Accuracy on training data: ', training_data_accuracy_rf)
print('Accuracy on test data: ', test_data_accuracy_rf)


# In[17]:


# Comparison of Accuracy Between Logistic Regression and Random Forest
models = ['Logistic Regression', 'Random Forest']
accuracies = [test_data_accuracy_lr, test_data_accuracy_rf]

plt.figure(figsize=(10, 6))
sns.barplot(x=models, y=accuracies, palette='viridis')
plt.title('Comparison of Accuracy Between Logistic Regression and Random Forest')
plt.ylabel('Accuracy')
plt.ylim(0, 1)

# Add accuracy values on top of each bar
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.01, f'{acc:.2f}', ha='center', va='bottom', fontsize=12)

plt.show()


# In[18]:


# Assuming 'new_data' contains 60 values for features\
new_data=(0.0119,0.0582,0.0623,0.0600,0.1397,0.1883,0.1422,0.1447,0.0487,0.0864,0.2143,0.3720,0.2665,0.2113,0.1103,0.1136,0.1934,0.4142,0.3279,0.6222,0.7468,0.7676,0.7867,0.8253,1.0000,0.9481,0.7539,0.6008,0.5437,0.5387,0.5619,0.5141,0.6084,0.5621,0.5956,0.6078,0.5025,0.2829,0.0477,0.2811,0.3422,0.5147,0.4372,0.2470,0.1708,0.1343,0.0838,0.0755,0.0304,0.0074,0.0069,0.0025,0.0103,0.0074,0.0123,0.0069,0.0076,0.0073,0.0030,0.0138)

new_data_array = np.array(new_data)

# Reshape the array to be a 2D array, as the model expects a 2D input
new_data_array_reshaped = new_data_array.reshape(1, -1)

# Scale the input data using the same scaler that was used for training
scaled_new_data = scaler.transform(new_data_array_reshaped)

# Now, you can make predictions using both models
rf_prediction = random_forest_model.predict(scaled_new_data)
lr_prediction = logistic_regression_model.predict(scaled_new_data)

# Print the predictions
print('Random Forest Prediction:', rf_prediction[0])


# In[ ]:




