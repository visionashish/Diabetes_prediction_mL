#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# In[19]:


df = pd.read_csv("diabetes.csv")
df


# In[20]:


missing_values=df.isnull().sum()
missing_values


# In[21]:


duplicate_values=df.duplicated().sum()
duplicate_values


# In[22]:


df_original = df.copy()
df_original


# In[23]:


print(df.dropna(inplace=True))
print(df.fillna(df.mode(),inplace=True))
print(df.drop_duplicates(inplace=True))


# In[24]:


print(df_original.shape)
print(df.shape)


# In[25]:


class_count = df['diabetes'].value_counts()
print(class_count)
class_count.plot(kind='bar')
plt.show()


# In[26]:


from imblearn.over_sampling import RandomOverSampler
x = df.drop('diabetes',axis = 1)
y = df['diabetes']

oversample = RandomOverSampler()
x_resample,y_resemble=oversample.fit_resample(x,y)
print(x_resample)
print(y_resemble)

df_balanced = pd.concat([x_resample,y_resemble],axis=1)


# In[27]:


count=df_balanced['diabetes'].value_counts()
count.plot(kind='bar')
plt.show()


# In[28]:


vars = ['gender','smoking_history']
#df_encoded = pd.get_dummies(df_balanced,columns=vars)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for v in vars:
    df_balanced[v] = le.fit_transform(df_balanced[v])
count=df_balanced['smoking_history'].value_counts()
count


# In[29]:


# Remove outliers using Z-score method
from scipy.stats import zscore

# Specify the numerical columns where you want to check for outliers
numerical_columns = df_balanced.select_dtypes(include=[np.number]).columns

# Calculate the Z-scores for the numerical columns
z_scores = np.abs(zscore(df_balanced[numerical_columns]))

# Define a threshold for Z-scores beyond which data points are considered outliers
z_threshold = 3

# Remove rows with outliers
df_no_outliers = df_balanced[(z_scores < z_threshold).all(axis=1)]

# Now df_no_outliers contains the data with outliers removed

# Continue with your analysis or modeling using df_no_outliers


# In[30]:


df_balanced


# In[31]:


df_no_outliers


# In[32]:


import seaborn as sns
def plot_box_plots(df, columns, title):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df[columns])
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()
    
plot_box_plots(df_balanced, numerical_columns, "Box Plots with Outliers")
plot_box_plots(df_no_outliers, numerical_columns, "Box Plots without Outliers")


# In[34]:


from sklearn.preprocessing import MinMaxScaler
x = df_no_outliers.drop('diabetes', axis=1)
print(x)
y = df_no_outliers['diabetes']
print(y)
scaler = MinMaxScaler()




# In[35]:


x_scaled= scaler.fit_transform(x)
df_scaled = pd.DataFrame(x_scaled,columns=x.columns)
df_scaled

print("Shape of df_scaled:", df_scaled.shape)
print("Shape of y:", y.shape)


# In[36]:


correlation_matrix = df_scaled.corr()
print(correlation_matrix)

print("Shape of df_scaled:", df_scaled.shape)
print("Shape of y:", y.shape)


# In[37]:


print(df_scaled.isnull().sum().sum())
print(y.isnull().sum())


# In[69]:


# Drop rows with missing values from both df_scaled and y


# Concatenate the cleaned dataframes
df_scaled_W_target = pd.concat([df_scaled,y], axis=1)

print("Shape of df_scaled_w_target:", df_scaled_W_target.shape)
print("Shape of y:", y.shape)




# In[70]:


# Calculate the correlation matrix
correlation_matrix = df_scaled_W_target.corr()


# Sort the correlation matrix by the absolute values of correlations with the target variable
correlation_with_target = abs(correlation_matrix['diabetes']).sort_values(ascending=False)

# Select the features with correlation above a certain threshold (e.g., 0.2)
threshold = 0.4
selected_features = correlation_with_target[correlation_with_target > threshold]

# Print the selected features and their correlation values
print(selected_features)


# In[71]:


selected_features=selected_features.drop('diabetes')
selected_features


# In[72]:


X=df_scaled[selected_features.index]
print(X)
y=df_scaled_W_target['diabetes']
print(y)



# In[75]:


y = y.dropna()


# In[76]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print("train set shape",X_train.shape,y_train.shape)
print("test set shape",X_test.shape,y_test.shape)


# In[83]:





# In[84]:


#from sklearn.linear_model import LogisticRegression
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.svm import SVC
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB





#from sklearn.naive_bayes import GaussianNB

#clf = LogisticRegression()
#clf = DecisionTreeClassifier()
#clf = RandomForestClassifier()
#clf = SVC()
#clf = KNeighborsClassifier()
clf = GaussianNB()

clf.fit(X_train,y_train)

y_train_pred = clf.predict(X_train)

y_test_pred = clf.predict(X_test)





# In[ ]:





# In[85]:


# Define the new data as a numpy array or a pandas DataFrame
new_data = np.array([[0.9, 0.1]])  # Replace with your own data

# Create a DataFrame (optional)
new_data_df = pd.DataFrame(new_data, columns=selected_features.index)  # Replace with your own column names
print(new_data_df)
# Apply the trained decision tree classifier to predict the labels for the new data
y_new_pred = clf.predict(new_data_df)

# Print the predicted labels for the new data
print(y_new_pred)



# In[ ]:





# In[ ]:





# In[86]:


from sklearn.metrics import classification_report

y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)



print("train Report : ",classification_report(y_train,y_train_pred))
print("test report : " ,classification_report(y_test,y_test_pred))


# In[87]:


from sklearn import tree
import matplotlib.pyplot as plt

# Create an instance of the DecisionTreeClassifier with max_depth=3
clf = tree.DecisionTreeClassifier(max_depth=2)

# Train the classifier on the training set
clf.fit(X_train, y_train)

# Display the decision tree

tree.plot_tree(clf, feature_names=X_train.columns, class_names=['No Diabetes', 'Diabetes'], filled=True)
plt.show()




import pickle
pickle.dump(clf, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(clf.predict([[0.9, 0.1]]))

