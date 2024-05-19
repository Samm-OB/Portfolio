#!/usr/bin/env python
# coding: utf-8

# This project is geared towards the problem of Fraud detection in the society. The challenges in using machine learning to solve fraud detection is 
# 1) Label Imbalance: In supervised Machine learning, the algorithms work best witha balanced dataset and this is most often not the case in real life.
# 2) Non-stationary data: The behavior of fraudsters quickly changes, which leads to changes in the data as well. This means that it’s important to constantly train new fraud detection models. One efficient way to do this is to set up a model retraining process to adapt faster and to catch fraudulent behavior much better.

# First we import the necessary modules/libraries

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# Firstly, I import the data and display part of it and perform some preprocessing actions which include:
# getting the list of columns, getting the no columns and size of the dataset, getting the dataset description, checking for and removing the no of empty columns or wrong values in columns, correlation of the columns withe each other

# In[4]:


df = pd.read_csv("Fraud.csv")
df.head()


# In[5]:


df.tail()


# In[6]:


df.columns


# brief description of some columns
# step : represents unit of time(i.e 1 step represents 1 hour of time),
# 
# type: method of transaction,
# 
# amount: amount of transaction,
# 
# oldbalaceorg : balance before transaction,
# 
# newbalanceorg : balance after transaction,
# 
# olbalancedest : initial balance of destination account,
# 
# newbalancedest : new balance of destination account

# In[7]:


df.size


# In[8]:


df.describe()


# From the above description
# 
# *The count of all the columns are the same so the data is balanced,
# 
# *The minimum amount in a transaction is '0' which may seem like the fraudsters have no idea of the amount in the accounts.

# In[9]:


#no of (rows, cols)
df.shape


# In[10]:


# to get what data types exist in each column
df.dtypes


# In[11]:


df[['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFraud',
       'isFlaggedFraud']].corr()


# From the above i can see that
# *There is multi-colinearity between the independent variables e.g(newbalanceorig, oldbalanceorig), (newbalancedest, oldbalancedest)
# 
# note:Multi-colinearity is the high correlation between 2 or more independent variables in a regression model. when independent variables correlate, this means they arer not truly independent and they move together in a way.
# 
# Effects include:
# 
# *The results/estimates may be vague, unreliable
# 
# *It becomes difficult to know how each independent variable individually influences the dependent variable

# we can find multi-colinearity using variation inflation value 

# In[12]:


#apply it


# In[13]:


df[df['amount'] == 0]


# All the transactions with '0' as the amount had '0' as their account balance and were fraud actions because it doesnt make sense to transfer '0' currency to a different account.

# In[14]:


df[df['isFlaggedFraud'] == 1]


# These may be transactions that were figured out and prevented before they occurred 

# In[15]:


df[(df['amount'] == df['oldbalanceOrg']) & (df['isFraud'] ==1)]


# If the amount in account is equal to the amount transferred then it is also an act of fraud and this means the fraudsters aimed at collecting all the money in the account

# In[16]:


df.isnull().sum()


# This means that all the columns contain values hence the dataset is clean

# In[17]:


df['isFraud'].value_counts()


# The Second step is Exploratory Data Analysis, this helps visualize distributions, corellations and outliers and understand patterns/relationships in  the data

# In[18]:


df.sample(10)


# In[19]:


df['type'].value_counts()


# In[20]:


df['isFraud'].value_counts()


# In[21]:


df['isFlaggedFraud'].value_counts()


# In[22]:


# to convert the integer values of the categories of the  "type" column of the dataset(i.e df['type'].value_counts) to a series
#This makes it easier to change the values to a list which would be used to calculate individual percentages of the pie chart

typeCatValues = pd.Series(df['type'].value_counts())
valList = typeCatValues.tolist()
valList


# between two variables. Scatter plots, bubble plots, and scatter plots with a line of best fit fall into this category1.
# 
# Histograms: Useful for understanding the distribution of continuous or categorical variables.
# 
# Box Plots: Show the distribution of data, including median, quartiles, and outliers.
# 
# Violin Plots: Combine box plots with kernel density estimation to display data distribution.
# 
# Time Series Plots: Depict how a value changes over time.
# 
# Area Charts: Show cumulative data trends.
# 
# Bar Charts: Useful for comparing categorical data.
# 
# Pie Charts: Display proportions of different categories.
# 
# Heatmaps: Visualize correlations or patterns in tabular data.
# 
# Treemaps: Represent hierarchical data structures.
# 
# Parallel Coordinates: Useful for visualizing multivariate data.
# 
# Dendrograms: Show hierarchical clustering relationships.
# 
# Density Plots: Illustrate the distribution of data points.
# 
# Lollipop Charts: Display differences between two points.
# 
# Waffle Charts: Depict proportions using squares or rectangles.

# In[21]:


# # Define payment method labels and colors
payment_method = ['CASH_OUT', 'PAYMENT', 'CASH_IN', 'TRANSFER', 'DEBIT']

#set colors to be used for Pie Chart
colors = ['red', 'green', 'blue', 'orange', 'purple']

new_df = {
    'type': valList,  # Example: [CASH_OUT, PAYMENT, CASH_IN, TRANSFER, DEBIT]
}

# wedge property is used to set the border of the pie chart (i.e Line width is for the thickness of circumference and edgecolor
# is to set the color )
wp = {'linewidth': 1, 'edgecolor': "black"}

#This is to seperate the pie chart
explode = (0.05, 0.07, 0.08, 0.08, 0.0)

# # Create a pie chart
def func(pct, allvalues):
    absolute = int(pct / 100.*np.sum(allvalues))
    return "{:.1f}%\n({:d} g)".format(pct, absolute)
 
fig, ax = plt.subplots(figsize=(10, 7))
wedges, texts, autotexts = ax.pie(new_df['type'],
                                  autopct=lambda pct: func(pct, new_df['type']),
                                  explode=explode,
                                  labels=payment_method,
                                  shadow=True,
                                  colors=colors,
                                  startangle=90,
                                  wedgeprops=wp, radius=1, rotatelabels=True)

# # Add legend
ax.legend(payment_method,
          title="Transaction Methods",
          loc="right",
          bbox_to_anchor=(1.5, 0.7, 0.3, 0.6))

# # Customize autotexts
plt.setp(autotexts, size=6, weight="bold")

# # Set title
ax.set_title("Transaction Methods", loc="left")

# # Show the plot
plt.show()


# From the Visualization it is clear that 'Cash-out' was the most used of transaction that occured followed by 'Payment'. This could mean that a good number of the fraudlent activities came from these 2 methods

# In[22]:


sns.heatmap(df[['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFraud',
       'isFlaggedFraud']].corr(), annot=True, cmap=sns.color_palette("Spectral", as_cmap=True), cbar=True)
plt.title("Correlation plot")
plt.show()


# This is the visualization of the correlation from before. There is a multi-correlation(i.e (newbalanceorig, oldbalanceorig), (newbalancedest, oldbalancedest)

# In[23]:


sns.barplot(x=df['type'], y=df['isFraud'], palette="Spectral")
plt.show()


# In[24]:


sns.countplot(x=df['isFraud'], palette="coolwarm")
plt.title("Unbalanced Dataset")
plt.show()


# In[173]:


figure = plt.subplots(figsize = (10, 6))

plt.subplot()
ax = sns.histplot(data=df[df['isFraud']==1],x='amount', palette="coolwarm")
plt.title('Amount of Fradulent Transactions for isFraud')

plt.show()


# # FEATURE EXTRACTION/ SELECTION

# I will be doing a feature extraction for the both supervised and unsupervised Learning methods where. Later i will be using K-Means(unsupervised algorithm), and  RandomForest(supervised Learning) algorithms, but the first I will first perform a supervised technique for the feature selection and PCA(unsupervised learning technique for feature selection)

# In[25]:


df['type'].unique()


# In[26]:


df[(df['type'] == "CASH_OUT") & (df['isFraud'] == 1)]


# In[27]:


df[(df['type'] == "CASH_OUT") & (df['isFraud'] == 0)]


# In[28]:


df[(df['type'] == "PAYMENT") & (df['isFraud'] == 0)]


# In[29]:


df[(df['type'] == "PAYMENT") & (df['isFraud'] == 1)]


# In[30]:


df[(df['type'] == "DEBIT") & (df['isFraud'] == 0)]


# In[31]:


df[(df['type'] == "DEBIT") & (df['isFraud'] == 1)]


# In[32]:


df[(df['type'] == "TRANSFER") & (df['isFraud'] == 0)]


# In[33]:


df[(df['type'] == "TRANSFER") & (df['isFraud'] == 1)]


# In[50]:


df[(df['type'] == "CASH_IN") & (df['isFraud'] == 1)]


# In[51]:


df[(df['type'] == "CASH_IN") & (df['isFraud'] == 0)]


# FROM THE ABOVE INFORMATION 'CASHOUT' HAS THE HIGHEST NO OF FRAUD ACTIVITIES FOLLOWED BY 'TRANSFER' WHILE THE REST HAVE NONE 

# In[52]:


df[['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFraud',
       'isFlaggedFraud']].corr()


# The "newbalanceorig" and "oldbalancedest" columns will be dropped because they are less correlated withe the dependent variable 
# "isFraud", The "namedest" and "nameorig" will be removed as well because they have too many unique values which would make is hard to encode unlike the "type" column that has 5 unique valuse that will be encoded

# # FEATURE SELECTION/EXTRACTION

# In[53]:


new_df = df.drop(['oldbalanceOrg'], axis=1)
new_df['type'] = new_df['type'].map({'PAYMENT': 1, 'TRANSFER': 2, 'CASH_OUT': 3, 'DEBIT': 4, 'CASH_IN': 5})
new_df = new_df.drop(['newbalanceOrig'], axis=1)
new_df = new_df.drop(['nameOrig'], axis=1)
new_df = new_df.drop(['nameDest'], axis=1)
new_df


# 
# #  BALANCING THE DATASET
# 

# In[54]:


normal_transactions = new_df[new_df['isFraud'] == 0]
fraud_transactions = new_df[new_df['isFraud'] == 1]
print(normal_transactions.shape, fraud_transactions.shape)


# The data is unbalanced just lke the visualsation from before so we have to normalize the data by making the no of fraud transactions and normal transactions equal

# In[39]:


normal_transactions = normal_transactions.sample(8213)


# In[40]:


normal_transactions.shape


# In[41]:


new_df = pd.concat([normal_transactions, fraud_transactions], axis=0)
new_df.head(15)


# In[42]:


new_df.tail(15)


# # TRAINING TESTING AND SPLITTING

# In this aspect i will be sepearting the features selected and splitting them into the training set(subdivided into training(0.6) and validation(0.2) sets)set and test set(0.2) because if it is not done like this then it will perform badly when deployed to the real world due to bad generalization. The validation set solves this problem
# 
# Methods of generating a validation include: Hold-out validation, Stratified Hold-out validation, k-fold Cross, and Leave one out Validation. But i will make use of Hold out validation and if after that each set does not get similar distribution(which would help generate a better model then) the stratified Hold-out validation will resolve this issue

# In[43]:


indep_varX = new_df.drop(['isFraud'], axis=1)
dep_varY= new_df['isFraud']


# In[44]:


from sklearn.model_selection import train_test_split as tts
train_new_dfX, test_new_dfX, train_new_dfY, test_new_dfY = tts(indep_varX, dep_varY, random_state=0, test_size=0.2, stratify=None)


# In[45]:


print("shape of train_new_dfX: ", train_new_dfX.shape)
print("shape of test_new_dfX: ", test_new_dfX.shape)
print("shape of train_new_dfY: ", train_new_dfY.shape)
print("shape of test_new_dfX: ", test_new_dfY.shape)


# In[46]:


from sklearn.model_selection import train_test_split as tts
train_x, val_x, train_y, val_y = tts(train_new_dfX, train_new_dfY, test_size=0.2, random_state=0, stratify=None)


# In[47]:


print("shape of training set: ", train_x.shape, train_y.shape)

print("shape of validation set: ", val_x.shape, val_y.shape)

print("shape of test set: ", test_new_dfX.shape, test_new_dfY.shape)


# In[48]:


train_y.value_counts()/ len(train_y)


# In[49]:


val_y.value_counts()/ len(val_y)


# In[50]:


train_new_dfY.value_counts()/ len(train_new_dfY)


# Since the distribution is slightly off then stratified hloud-out method will be used. though the issue with this method is we wont get enough training data but the solution to this k-fold cross

# In[130]:


from sklearn.model_selection import train_test_split as tts
train_new_dfX, test_new_dfX, train_new_dfY, test_new_dfY = tts(indep_varX, dep_varY, random_state=0, test_size=0.2, stratify=dep_varY)


# In[131]:


from sklearn.model_selection import train_test_split as tts
train_x, val_x, train_y, val_y = tts(train_new_dfX, train_new_dfY, test_size=0.2, random_state=42, stratify=train_new_dfY)


# In[132]:


print("shape of training set: ", train_x.shape, train_y.shape)

print("shape of validation set: ", val_x.shape, val_y.shape)

print("shape of test set: ", test_new_dfX.shape, test_new_dfY.shape)


# In[133]:


train_y.value_counts()/ len(train_y)


# In[134]:


val_y.value_counts()/ len(val_y)


# In[135]:


train_new_dfY.value_counts()/ len(train_new_dfY)


# SCALING

# Data scaling is a method for reducing the effect of data bias on predictions which is highly used in pre-processing step in any Machine Learning project and they are of 3 types: Standard Scaler, Min-Max Scaler and Robust Scaler
# 
# But The Robust Scaler will be used for this model due to its ability to detect outliers unlike the standard scaler and its only weakness s that it does not take into account the mean and median because it uses quartiles and median to tackle biases from outliers.

# In[184]:


from sklearn.preprocessing import RobustScaler


# In[185]:


scaler = RobustScaler()


# In[186]:


scaler.fit(train_x)


# In[187]:


trainX_scaler = scaler.transform(train_x)


# In[188]:


testX_scaler = scaler.transform(test_new_dfX)


# In[189]:


robust_df = pd.DataFrame(trainX_scaler, columns =['step', 'type', 'amount', 'oldbalanceDest', 'newbalanceDest', 
                                                  'isFlaggedFraud'])


# In[190]:


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have train_x, train_y, test_new_dfX, and test_new_dfY defined
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 5))

# Plotting the KDE plot for 'type' before scaling
sns.kdeplot(train_x['newbalanceDest'], ax=ax1, color='r')
ax1.set_title('Before Scaling')

# Plotting the KDE plot for 'x2' after robust scaling
sns.kdeplot(train_x['oldbalanceDest'], ax=ax1, color='b')

ax2.set_title('After Robust Scaling')
sns.kdeplot(robust_df['newbalanceDest'], ax = ax2, color ='green')
sns.kdeplot(robust_df['oldbalanceDest'], ax = ax2, color ='black',)
plt.show()


# From the plot we see that after robust scaling we have a perfectly fitted feature ("oldbalancedest", newbalancedest)

# In[191]:


trainX_scaler


# In[192]:


testX_scaler


# # TRAINING AND EVALUATING MODEL

# Using the Logistic Regression, Ths can be used for fraud detection, Disease prediction or churn prediction

# In[193]:


from sklearn.linear_model import LogisticRegression


# In[194]:


logReg_model = LogisticRegression()


# In[195]:


logReg_model.fit(trainX_scaler, train_y)


# METRIC AND ACCURACY

# In[196]:


#TO TEST THE ACCURACY OF THE MODEL
from sklearn.metrics import accuracy_score


# In[200]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
pred_y = logReg_model.predict(testX_scaler)
f1 = f1_score(test_new_dfY.values, pred_y)
roc_auc = roc_auc_score(test_new_dfY.values, pred_y)
conf_matrix = confusion_matrix(test_new_dfY.values, pred_y)


# In[203]:


print("-----------------------------------------")

print(f"Accuracy :{accuracy_score(test_new_dfY.values, pred_y)*100}%")

print("-----------------------------------------")

print("-----------------------------------------")
print(f'F1 Score: {f1}')
print("-----------------------------------------")

print("-----------------------------------------")
print(f'ROC AUC: {roc_auc}')
print("-----------------------------------------")

print("-----------------------------------------")
print(f'Confusion Matrix:\n{conf_matrix}')
print("-----------------------------------------")


# Using Random forest Classifier

# In[205]:


from sklearn.ensemble import RandomForestClassifier


# In[206]:


randFor_model = RandomForestClassifier()


# In[207]:


randFor_model.fit(trainX_scaler, train_y)


# In[208]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
pred_y = randFor_model.predict(testX_scaler)
f1 = f1_score(test_new_dfY.values, pred_y)
roc_auc = roc_auc_score(test_new_dfY.values, pred_y)
conf_matrix = confusion_matrix(test_new_dfY.values, pred_y)


# In[209]:


print("-----------------------------------------")

print(f"Accuracy :{accuracy_score(test_new_dfY.values, pred_y)*100}%")

print("-----------------------------------------")

print("-----------------------------------------")
print(f'F1 Score: {f1}')
print("-----------------------------------------")

print("-----------------------------------------")
print(f'ROC AUC: {roc_auc}')
print("-----------------------------------------")

print("-----------------------------------------")
print(f'Confusion Matrix:\n{conf_matrix}')
print("-----------------------------------------")

