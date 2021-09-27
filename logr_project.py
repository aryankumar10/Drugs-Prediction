'''

    Project on correct predicting drugs for patient
    using Logistic Regresstion model
    
    Created by : Aryan Kumar

'''

#===================================
# Importing important libraries
#===================================
import os
import sys
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Function to encode data
def label_change(df, cols):
    le = LabelEncoder()
    for i in cols:
        if df[i].dtype == 'object':
            df[i] = le.fit_transform(df[i])

# Function to return a key of a value in dictionary
def get_key(refer, val):
    for key, value in refer.items():
        for element in value:
            if val == element:
                return key

# Function to assign key to a data
def assign(refer, val):
    for key, value in refer.items():
        for element in value:
            if val == element:
                val = get_key(refer, val)
                return val

os.chdir('C:/Users/aryan/OneDrive/Desktop/project')
data_original = pd.read_csv('drug.csv')
data = data_original.copy()

# Checking if there are any null values in the dataset
#data.isnull().sum()

# Printing the information about the dataset
#data.info()
# The target variable y is 'Drug'

#=================================================
# Replacing alphabetical values with numerical
#=================================================
cols = list(data.columns[1:])
label_change(data, cols)

#===============================================
# Splitting data into test set and train set
#===============================================
x = data.drop(['Drug'], axis=1)
y = data['Drug']

# test set is 30% and train set is 70% data of the original dataset
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=0)

#===============================================================
# Fitting the data to Logistic Regression instance - logistic
#===============================================================
logistic = LogisticRegression(max_iter=5000)
logistic.fit(train_x, train_y)
prediction = logistic.predict(test_x)
acc = accuracy_score(test_y, prediction)
print("The accuracy of the model is : {} %".format(acc*100))

#===================
# Plotting graphs
#===================
#sns.boxplot('Na_to_K','Drug',data=data_original)
#sns.boxplot('Age','Drug',data=data_original)
'''
#==========================================================
# Creating various lists and dictionaries for user input
#==========================================================
list_sex = ['M', 'm', 'F', 'f']
list_bp = ['HIGH', 'High', 'high', 'LOW', 'Low', 'low', 'NORMAL', 'Normal', 'normal']
list_chol = ['HIGH', 'High', 'high', 'NORMAL', 'Normal', 'normal']

refer_features = {0: list_bp[:3]+list_sex[:2]+list_chol[:3],
                  1: list_bp[3:6]+list_sex[2:]+list_chol[3:],
                  2: list_bp[6:]}

refer_drugs = {'DrugY': [0],
               'drugA': [1],
               'drugB': [2],
               'drugC': [3],
               'drugX': [4]}

#=======================================
# Inputting new data input from user
#=======================================
print("\t\t*MEDICAL DRUG ANALYSIS*")

# Now we input new values and predict a drug for patient
try :
    age = int(input("Enter Age: "))
    if age<0 :
        sys.exit()

    sex = (input("Enter gender (M/F): "))
    if sex not in list_sex:
        sys.exit()

    bp = (input("Enter BP (HIGH/LOW/NORMAL): "))
    if bp not in list_bp:
        sys.exit()

    chol = (input("Enter Cholesterol (HIGH/NORMAL): "))
    if chol not in list_chol:
        sys.exit()

    na_to_k = float(input("Enter Na to K ratio: "))
    if na_to_k<0:
        sys.exit()

    # Converting user input data to acceptable input
    new_data = pd.DataFrame({"Age":[age], "Sex":[sex], "BP":[bp], "Cholesterol":[chol], "Na_to_K":[na_to_k]})
    new_data["Sex"].replace(sex, assign(refer_features, sex), inplace=True)
    new_data["BP"].replace(bp, assign(refer_features, bp), inplace=True)
    new_data["Cholesterol"].replace(chol, assign(refer_features, chol), inplace=True)
    
    # Predicting drug for new user input
    new_prediction = logistic.predict(new_data)
    new_drug = get_key(refer_drugs, new_prediction)
    print("The drug prescribed should be: ",new_drug)
    
except:
    print("PLEASE ENTER VALID INPUT")

'''



