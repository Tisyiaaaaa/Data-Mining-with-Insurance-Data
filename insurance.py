#!/usr/bin/env python
# coding: utf-8

# **THING TO REMEBER: PURCHASED PLAN MEANS THE CURRENT POLICY THE CUSTOMER CURRENTLY HAVING (PAST POLICY THEY HAVE BOUGHT) ; NEEDED_PLAN IS WHAT THE CUSTOMER WANTS TO BUY FOR NOW AND FUTURE (RENEW THEIR EXISTING PLAN (PURCHASED PLAN) )**

# In[1]:

heroku/python

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import missingno as msno
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import datetime as dt  
import streamlit as st
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import random

import warnings
warnings.filterwarnings('ignore')

import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
sns.set(rc={'figure.figsize':(11,6)})

from scipy.stats import spearmanr #spearmanr is used to find correlation

import missingno as msno

pd.set_option('display.max_columns', 500)


# In[2]:


df = pd.read_csv("insurance.csv")
df


# In[3]:


nulls= df.isnull().sum()
nulls[nulls > 0]


# In[4]:


df['FamilyExpenses(month)']=df['FamilyExpenses(month)'].fillna(df['FamilyExpenses(month)'].mean())
df['AnnualSalary']=df['AnnualSalary'].fillna(df['AnnualSalary'].mean())
df['Age']=df['Age'].fillna(df['Age'].median())


# In[5]:


nulls= df.isnull().sum()
nulls[nulls > 0]


# In[6]:


df.Age.describe()


# In[7]:


df['MaritalStatus'] =df['MaritalStatus'].fillna("Unknown")
df['SmokerStatus'] =df['SmokerStatus'].fillna("Non_Smoker")
df['Race'] =df['Race'].fillna("Unknown")
df['Occupation'] =df['Occupation'].fillna("Unemployed")
df['Nationality'] =df['Nationality'].fillna("Unknown")
df['HomeAddress'] =df['HomeAddress'].fillna("central_mal")
df['NoOfDependent'] =df['NoOfDependent'].fillna(0)


# In[8]:


dle=df.copy()
le = preprocessing.LabelEncoder()
df["Gender"] = le.fit_transform(dle["Gender"].astype(str))
#df["SmokerStatus"] = le.fit_transform(dle["SmokerStatus"].astype(str))
dle["LifeStyle"] = le.fit_transform(dle["LifeStyle"].astype(str))
dle["MaritalStatus"] = le.fit_transform(dle["MaritalStatus"].astype(str))
dle["LanguageSpoken"] = le.fit_transform(dle["LanguageSpoken"].astype(str))
dle["HighestEducation"] = le.fit_transform(dle["HighestEducation"].astype(str))
dle["Race"] = le.fit_transform(dle["Race"].astype(str))
dle["Nationality"] = le.fit_transform(dle["Nationality"].astype(str))
dle["MalaysiaPR"] = le.fit_transform(dle["MalaysiaPR"].astype(str))
dle["MovingToNewCompany"] = le.fit_transform(dle["MovingToNewCompany"].astype(str))
#dle["Occupation"] = le.fit_transform(dle["Occupation"].astype(str))
dle["Telco"] = le.fit_transform(dle["Telco"].astype(str))
dle["HomeAddress"] = le.fit_transform(dle["HomeAddress"].astype(str))
#dle["ResidentialType"] = le.fit_transform(dle["ResidentialType"].astype(str))
#dle["Customer_Needs_1"] = le.fit_transform(dle["Customer_Needs_1"].astype(str))
#dle["Customer_Needs_2"] = le.fit_transform(dle["Customer_Needs_2"].astype(str))
#dle["PurchasedPlan1"] = le.fit_transform(dle["PurchasedPlan1"].astype(str))
dle["Transport"] = le.fit_transform(dle["Transport"].astype(str))
#dle["PurchasedPlan2"] = le.fit_transform(dle["PurchasedPlan2"].astype(str))
#dle["MedicalComplication"] = le.fit_transform(dle["MedicalComplication"].astype(str))


# In[9]:


from sklearn.preprocessing import LabelEncoder
encoder =  LabelEncoder()

insure_sex =  dle["Gender"]
#insure_smoker =  dle["SmokerStatus"]
insure_lifestyle =  dle["LifeStyle"]
insure_marital =  dle["MaritalStatus"]
insure_language =  dle["LanguageSpoken"]
insure_highedu =  dle["HighestEducation"]
insure_race =  dle["Race"]
insure_nationality =  dle["Nationality"]
insure_masPR = dle["MalaysiaPR"]
insure_movingnewcompany = dle["MovingToNewCompany"]
#insure_occupation = dle["Occupation"]
insure_telco = dle["Telco"]
insure_address = dle["HomeAddress"]
#insure_ResType = dle["ResidentialType"]
#insure_cus1 = dle["Customer_Needs_1"]
#insure_cus2 = dle["Customer_Needs_2"]
#insure_purplan1 = dle["PurchasedPlan1"]
insure_transport = dle["Transport"]
#insure_purplan2 = dle["PurchasedPlan2"]
#insure_medical = dle["MedicalComplication"]

insure_sex_encoded = encoder.fit_transform(insure_sex)
#insure_smoker_encoded = encoder.fit_transform(insure_smoker)
insure_region_encoded = encoder.fit_transform(insure_lifestyle)
insure_marital_encoded = encoder.fit_transform(insure_marital)
insure_language_encoded = encoder.fit_transform(insure_language)
insure_highedu_encoded = encoder.fit_transform(insure_highedu)
insure_race_encoded = encoder.fit_transform(insure_race)
insure_nationality_encoded = encoder.fit_transform(insure_nationality)
insure_masPR_encoded = encoder.fit_transform(insure_masPR)
insure_movingnewcompany_encoded = encoder.fit_transform(insure_movingnewcompany)
#insure_occupation_encoded = encoder.fit_transform(insure_occupation)
insure_telco_encoded = encoder.fit_transform(insure_telco)
insure_address_encoded = encoder.fit_transform(insure_address)
#insure_ResType_encoded = encoder.fit_transform(insure_ResType)
#insure_cus1_encoded = encoder.fit_transform(insure_cus1)
#insure_cus2_encoded = encoder.fit_transform(insure_cus2)
#insure_purplan1_encoded = encoder.fit_transform(insure_purplan1)
insure_transport_encoded = encoder.fit_transform(insure_transport)
#insure_purplan2_encoded = encoder.fit_transform(insure_purplan2)
#insure_medical_encoded = encoder.fit_transform(insure_medical)


# In[10]:


dfa = df.copy()

# df2['age_bins'] = pd.cut(x=df2['Age_Range'], bins=[28,35,36,43,44,51,52,59], labels=['Young_Adult','Mid_Aged_Adult','Older_Adult','Senior_Citizen'])
dfa['age_bins'] = pd.cut(x=dfa['Age'], bins=[17,25,33,41, 48], labels=['Teenager','Young_Adult','Mid_Aged_Adult','Older_Adult'])


# In[11]:


dfa


# customer moslty are smoker

# In[12]:


b=sns.countplot(x='SmokerStatus', data = dfa)

for p in b.patches:
        b.annotate("%.0f" % p.get_height(), (p.get_x() + 
    p.get_width() / 2., p.get_height()), 
        ha='center', va='center', rotation=0, 
    xytext=(0, 18), textcoords='offset points')
        
st.show(b)


# this is ok, seems like the 'frequent smoker' buy medical plan, so we do not have to suggest smoker customer to buy medical plan

# In[13]:


a=pd.crosstab(dfa.SmokerStatus,dfa.Customer_Needs_1).plot(kind='barh')
plt.title('Stacked Bar Chart of SmokerStatus vs Customer_Needs_1')
plt.xlabel('SmokerStatus')
plt.ylabel('Customer_Needs_1')
st.show(a)


# this must ignore the 0.0 as the 0.0 is just we replace it as NaN value, we do not know actually is what number.

# this seems like doesnt indicate anything. can consider delete this

# In[14]:


a=pd.crosstab(dfa.PurchasedPlan1,dfa.NoOfDependent).plot(kind='barh')
plt.title('Stacked Bar Chart of PurchasedPlan1 vs NoOfDependent')
plt.xlabel('PurchasedPlan1')
plt.ylabel('NoOfDependent')
st.show(a)


# could suggest older adult to purchase NoMoneyDown (invesment) to concern about their family financial if they have passed away, they still have this invesment money for their family. Could also suggest them to buy NoMoneyDown (invesment) as their new hobby after retirement. 

# for kidsflyup it seems like ok cause young adult is 25-33, their kids still small, so buy this plan is suitable, so for fkidsflyup no problem

# for xedu is seems ok also cause young_adult and mid age adult buy for their kids, teenage buy for themselve

# In[15]:


a=pd.crosstab(dfa.PurchasedPlan2,dfa.age_bins).plot(kind='barh')
plt.title('Stacked Bar Chart of PurchasedPlan2 vs Age_Bins')
plt.xlabel('PurchasedPlan2')
plt.ylabel('Age_Bins')
st.show(a)


# Self-employed people should buy persoanl retirement as they do not have epf

# Un-employed people should buy personal saving as they do not have income, they at least still have saving to maintain their daily life

# In[16]:


table=pd.crosstab(dfa.Occupation, dfa.Customer_Needs_1)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Occupation vs Customer_Needs_1')
plt.xlabel('Occupation')
plt.ylabel('Customer_Needs_1')
st.show(table)


# In[17]:


a=pd.crosstab(dfa.Occupation,dfa.Customer_Needs_1).plot(kind='barh')
plt.title('Stacked Bar Chart of Occupation vs Customer_Needs_1')
plt.xlabel('Occupation')
plt.ylabel('Customer_Needs_1')
st.show(a)


# Un-employed people should buy invesment(No Money Down) as their side-income when they do not have any job

# In[18]:


a=pd.crosstab(dfa.Occupation,dfa.PurchasedPlan2).plot(kind='barh')
plt.title('Stacked Bar Chart of Occupation vs PurchasedPlan2')
plt.xlabel('Occupation')
plt.ylabel('PurchasedPlan2')
st.show(a)


# In[19]:


dfa.groupby('age_bins').sum()
gkk = dfa.groupby(['Customer_Needs_1', 'PurchasedPlan2']) 

c = gkk.age_bins.value_counts().plot(kind = 'barh', figsize =(20,20), stacked=True)
st.show(c)


# # ONE HOT & FEATURE SELECTION

# In[20]:


def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))


# In[21]:


df["MedicalComplication"] = le.fit_transform(df["MedicalComplication"].astype(str))
insure_medical = df["MedicalComplication"]
insure_medical_encoded = encoder.fit_transform(insure_medical)


# In[22]:


col_list = [col for col in df.columns.tolist() if df[col].dtype.name == "object"]
df_oh = df[col_list]
df = df.drop(col_list, 1)

#the only needed code in this
df_oh = pd.get_dummies(df_oh)


df = pd.concat([df, df_oh], axis=1)
df.head()


# In[23]:


y = df.MedicalComplication
X = df.drop("MedicalComplication", 1)
colnames = X.columns


# # BORUTA 

# In[24]:


from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
rf = RandomForestClassifier(n_jobs=-1,class_weight="balanced",max_depth=5)
fast_selector = BorutaPy(rf, n_estimators="auto",random_state =1)


# In[25]:


fast_selector.fit(X.values,y.values.ravel())


# In[26]:


from sklearn.preprocessing import MinMaxScaler
boruta_score = ranking(list(map(float,fast_selector.ranking_)),colnames,order=-1)
boruta_score = pd.DataFrame(list(boruta_score.items()),columns=["Features","Score"])
boruta_score = boruta_score.sort_values("Score",ascending=False)  
     


# In[27]:


print('---------Top 10----------')
display(boruta_score.head(10))

print('---------Bottom 10----------')
boruta_score.tail(10)


# In[28]:


sns_boruta_plot = sns.catplot(x="Score", y="Features", data = boruta_score[0:30], kind = "bar", 
               height=14, aspect=1.9, palette='coolwarm')
plt.title("Boruta Top 30 Features")
st.show(sns_boruta_plot)


# # RFE

# In[29]:


# from sklearn.feature_selection import RFECV
# rf = RandomForestClassifier(n_jobs=-1,class_weight="balanced",max_depth=5,n_estimators=100)
# rf.fit(X,y)
# rfe = RFECV(rf, min_features_to_select=1,cv=3)


# In[30]:


# rfe.fit(X,y)


# In[31]:


# rfe_score = ranking(list(map(float, rfe.ranking_)), colnames, order=-1)
# rfe_score = pd.DataFrame(list(rfe_score.items()), columns=['Features', 'Score'])
# rfe_score = rfe_score.sort_values("Score", ascending = False)


# In[32]:


# sns_rfe_plot = sns.catplot(x="Score", y="Features", data = rfe_score[0:30], kind = "bar", 
#                height=14, aspect=1.9, palette='coolwarm')
# plt.title("RFE Top 30 Features")


# # DUMMY 

# In[33]:


# dle["MedicalComplication"] = le.fit_transform(dle["MedicalComplication"].astype(str))
# insure_medical = dle["MedicalComplication"]
# insure_medical_encoded = encoder.fit_transform(insure_medical)


# In[34]:


# dummy = dle[['SmokerStatus','Occupation',"Customer_Needs_1", 'Customer_Needs_2', 'PurchasedPlan1', 'PurchasedPlan2', 'MedicalComplication']]
# dummy


# In[35]:


# cat_vars=['SmokerStatus','Occupation',"Customer_Needs_1", 'Customer_Needs_2', 'PurchasedPlan1', 'PurchasedPlan2']
# for var in cat_vars:
#     cat_list='var'+'_'+var
#     cat_list = pd.get_dummies(dummy[var],prefix=var)
#     df1=dummy.join(cat_list)
#     dummy=df1
# df_vars = dummy.columns.values.tolist()
# to_keep=[i for i in df_vars if i not in cat_vars]


# In[36]:


# df_final=dummy[to_keep]
# df_final.columns.values


# In[37]:


# X = df_final.loc[:, df_final.columns != 'MedicalComplication']
# y = df_final.loc[:, df_final.columns == 'MedicalComplication']

# from imblearn.over_sampling import SMOTE #to make the data balance

# # SMOTE codes here...
# os=SMOTE(random_state=0)
# X_train, X_test, y_train, y_test = train_test_split(X,y.values.ravel(),test_size=0.3, random_state=0)
# columns = X_train.columns
# os_data_X,os_data_y=os.fit_sample(X_train,y_train)
# os_data_X = pd.DataFrame(data=os_data_X,columns=columns)
# os_data_y = pd.DataFrame(data=os_data_y,columns=['MedicalComplication'])

# print("length of oversampled data is ", len (os_data_X))
# print("Number of no subscription in oversampled data", len(os_data_y[os_data_y['MedicalComplication']==0]))
# print("Number of subscription ", len(os_data_y[os_data_y['MedicalComplication']==1]))
# print("Proportion of no subscription data in oversampled data is ", len(os_data_y[os_data_y['MedicalComplication']==0])/len(os_data_X))
# print("Proportion of subscription data in oversampled data is ", len(os_data_y[os_data_y['MedicalComplication']==1])/len(os_data_X))


# In[38]:


# os_data_X,os_Data_y=os.fit_sample(X_train,y_train)


# In[39]:


# import statsmodels.api as sm
# logit_model=sm.Logit(y,X)
# result=logit_model.fit(maxiter=400)
# print(result.summary2())


# In[40]:


# nb = dfa[['Occupation','FamilyExpenses(month)','AnnualSalary']]
# nbb = df[['Occupation_Unemployed','Occupation_employer','Occupation_govServant','Occupation_privateEemployee','FamilyExpenses(month)','AnnualSalary']]


# In[41]:


# # For this example, we use the mass, width, and height features of each fruit instance
# X1 = nbb.drop('Occupation_Unemployed', axis=1)
# y1 = nbb['Occupation_Unemployed']
# X2 = nbb.drop('Occupation_employer', axis=1)
# y2 = nbb['Occupation_employer']
# X3 = nbb.drop('Occupation_govServant', axis=1)
# y3 = nbb['Occupation_govServant']
# X4 = nbb.drop('Occupation_privateEemployee', axis=1)
# y4 = nbb['Occupation_privateEemployee']
# X = nb.drop('Occupation', axis=1)
# y = nb['Occupation']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=0)
# X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, random_state=0)
# X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.3, random_state=0)
# X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size=0.3, random_state=0)


# In[42]:


# from sklearn.naive_bayes import GaussianNB

# nb1= GaussianNB()
# nb1.fit(X1_train, y1_train)
# nb2= GaussianNB()
# nb2.fit(X2_train, y2_train)
# nb3= GaussianNB()
# nb3.fit(X3_train, y3_train)
# nb4= GaussianNB()
# nb4.fit(X4_train, y4_train)
# nb= GaussianNB()
# nb.fit(X_train, y_train) #this is for df b4 lbe 


# In[43]:


# print('Accuracy of unemployed= {:.2f}'. format(nb1.score(X1_test, y1_test)))
# print('Accuracy of employer= {:.2f}'. format(nb2.score(X2_test, y2_test)))
# print('Accuracy of goverment servant= {:.2f}'. format(nb3.score(X3_test, y3_test)))
# print('Accuracy of of private employee= {:.2f}'. format(nb4.score(X4_test, y4_test)))
# print('Accuracy of of occupation b4 applying lbe= {:.2f}'. format(nb.score(X_test, y_test)))

