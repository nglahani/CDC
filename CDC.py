#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# '''Pandas and Numpy imports for DataFrame'''
import pandas as pd
import numpy as np

#'''Seaborn and Matplotlib Visualization'''
import matplotlib                  # 2D Plotting Library
import matplotlib.pyplot as plt
import seaborn as sns              # Python Data Visualization Library based on matplotlib
plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')

#'''Plotly Visualizations'''
import plotly as plotly                # Interactive Graphing Library for Python
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot, plot
import plotly.offline as py
init_notebook_mode(connected=True)
import os
get_ipython().run_line_magic('pylab', 'inline')
import re

import warnings
warnings.filterwarnings('ignore')


# In[2]:


#Reads in CSV File downloaded from Database
data = pd.read_csv('CDC_CLEANED.csv')


# In[3]:


data.dropna(inplace = True, thresh = 250000, axis = 1)


# In[4]:


data.drop(['open-end_line_of_credit', 'rate_spread', 'total_loan_costs', 'origination_charges',
        'loan_term', 'interest_only_payment',
       'co-applicant_credit_score_type',
       'applicant_ethnicity-1', 'co-applicant_ethnicity-1', 'applicant_race-1',
       'co-applicant_race-1', 'co-applicant_age','applicant_age_above_62'], inplace = True, axis =1)


# In[5]:


data.drop(['tract_owner_occupied_units', 'tract_one_to_four_family_homes','tract_median_age_of_housing_units','tract_population','total_units','business_or_commercial_purpose','preapproval','purchaser_type','action_taken','total_units','tract_population','census_tract']
          , axis =1, inplace = True)


# In[6]:


#Unwanted Columns Dropped
data.head()


# In[7]:


# of Number of total Rows and Columns in DataFrame
data.shape


# In[8]:


data.dtypes


# In[9]:


#Data Cleaned and seperated in order to remove null values and unavailable entries

genderdata=data[(data["derived_sex"] != 'Sex Not Available')]
racedata = data[(data["derived_race"] != 'Race Not Available') & (data["derived_race"] != 'Free Form Text Only')]
ethnicdata = data[(data["derived_ethnicity"] != 'Ethnicity Not Available') & (data["derived_ethnicity"] != 'Free Form Text Only')]


# In[10]:


#SeaBorn BoxPlot: derived_race vs. loan_amount

plt.rcParams['figure.figsize'] = (18,10)
sns.boxplot(x = 'derived_race', y = 'loan_amount', data = racedata, showfliers = False)
plt.xticks(rotation = 45, size = 13)


# In[11]:


#SeaBorn BoxPlot: derived_ethnicity vs. loan_amount

loanvethdata=data[(data["derived_ethnicity"] != 'Sex Not Available')]
plt.rcParams['figure.figsize'] = (18,10)
sns.boxplot(x = 'derived_ethnicity', y = 'loan_amount', data = ethnicdata, showfliers = False)
plt.xticks(rotation = 45, size = 13)


# In[12]:


#Seaborn: Age Distribution

f,ax=plt.subplots(1,2,figsize=(18,8))

genderdata['derived_sex'].value_counts().plot.pie(explode=[0,0.05,0],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Distribution of Gender')

sns.countplot('derived_sex',data=genderdata,ax=ax[1])
ax[1].set_title('Distribution of Gender')
plt.show()


# In[13]:


#Seaborn: Age distribution

data['min_age'] = data['applicant_age'].apply(lambda x: x.split('-')[0])
data['max_age'] = data['applicant_age'].apply(lambda x: x.split('-')[-1])
data['min_age'] = data['min_age'].apply(lambda x: x.replace('>','').replace('<',''))
data['max_age'] = data['max_age'].apply(lambda x: x.replace('>','').replace('<',''))
data['max_age'] = pd.to_numeric(data['max_age'])
data['min_age'] = pd.to_numeric(data['min_age'])
data['avg_age'] = (data['max_age'] + data['min_age'])/2

#filter out 8888.0 and 9999.0 years old rows
data = data[(data['avg_age']!=8888.0) & (data['avg_age']!= 9999.0)]

plt.rcParams['figure.figsize'] = (18,8)
plt.xticks(size = 15)
plt.xlabel('AGE',size = 20)
plt.ylabel('COUNT', size = 20)
plt.title('Distribution of AGE', size = 25)
sns.countplot(data['avg_age'])


# In[14]:


#Plotly Derived Race Distributions

labels = sorted(racedata.derived_race.unique())
values = racedata.derived_race.value_counts().sort_index()
colors = ['DarkGrey', 'HotPink']



fig = go.Figure(data=[go.Pie(labels=labels,
                             values=values, pull=[0, 0.06])])
fig.update_traces(hoverinfo='label+value', textinfo='percent',textfont_size=20,
                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))

fig.update_layout(title_text="Distribution of Derived Race")
fig.show()


# In[15]:


#Plotly: Derived Ethnicity Distribution

labels = sorted(ethnicdata.derived_ethnicity.unique())
values = ethnicdata.derived_ethnicity.value_counts().sort_index()
colors = ['DarkGrey', 'HotPink']


fig = go.Figure(data=[go.Pie(labels=labels,
                             values=values, pull=[0, 0.06])])
fig.update_traces(hoverinfo='label+value', textinfo='percent',textfont_size=20,
                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))

fig.update_layout(title_text="Distribution of Derived Ethnicity")
fig.show()


# In[18]:


#Plotly: Conforming Loan Limit - Gender

ccl_c = genderdata[genderdata['conforming_loan_limit'] == 'C']
ccl_nc = genderdata[genderdata['conforming_loan_limit'] == 'NC']
ccl_u = genderdata[genderdata['conforming_loan_limit'] == 'U']

trace1 = go.Histogram(
    x=ccl_c.derived_sex,
    opacity=0.85,
    name = "Conforming",
    marker=dict(color='LightSeaGreen',line=dict(color='#000000', width=2)))
trace2 = go.Histogram(
    x=ccl_nc.derived_sex,
    opacity=0.85,
    name = "Not Conforming",
    marker=dict(color='Crimson',line=dict(color='#000000', width=2)))
trace3 = go.Histogram(
    x=ccl_u.derived_sex,
    opacity=0.85,
    name = "Undeterminate",
    marker=dict(color='Coral',line=dict(color='#000000', width=2)))

plotdata = [trace1, trace2, trace3]
layout = go.Layout(barmode='stack',
                   title='Conforming Loan Limit - GENDER',
                   xaxis=dict(title='GENDER'),
                   yaxis=dict( title='COUNT'),
                   template = 'plotly_dark'
)
fig = go.Figure(data=plotdata, layout=layout)
iplot(fig)


# In[19]:


#Plotly: Conforming Loan Limit - Derived Race

ccl_c = racedata[racedata['conforming_loan_limit'] == 'C']
ccl_nc = racedata[racedata['conforming_loan_limit'] == 'NC']
ccl_u = racedata[racedata['conforming_loan_limit'] == 'U']

trace1 = go.Histogram(
    x=ccl_c.derived_race,
    opacity=0.85,
    name = "Conforming",
    marker=dict(color='DodgerBlue',line=dict(color='#000000', width=2)))
trace2 = go.Histogram(
    x=ccl_nc.derived_race,
    opacity=0.85,
    name = "Not Conforming",
    marker=dict(color='Crimson',line=dict(color='#000000', width=2)))
trace3 = go.Histogram(
    x=ccl_u.derived_race,
    opacity=0.85,
    name = "Undeterminate",
    marker=dict(color='Coral',line=dict(color='#000000', width=2)))



plotdata = [trace1, trace2, trace3]
layout = go.Layout(barmode='stack',
                   title='Conforming Loan Limit - RACE',
                   xaxis=dict(title='RACE'),
                   yaxis=dict( title='COUNT'),
                   template = 'plotly_dark'
)
fig = go.Figure(data=plotdata, layout=layout)
iplot(fig)


# In[20]:


#Feature Transformation and Scaling  

new_data = data[pd.notnull(data['county_code'])]
fin_data = new_data.copy()


# In[21]:


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector as selector

numeric_features = ['income', 'loan_amount', 'tract_minority_population_percent']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['applicant_age', 'derived_sex', 'derived_race', 'derived_ethnicity', 'loan_type', 'county_code', 'denial_reason-1']

categorical_transformer = Pipeline(steps=[
    ('encoder', OrdinalEncoder()),
    #('imputer', SimpleImputer(strategy='constant', fill_value='mode')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
     ])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, selector(dtype_exclude='object')),
    ('cat', categorical_transformer, selector(dtype_include='object'))
    ])

total_features = ['income', 'loan_amount', 'tract_minority_population_percent', 'applicant_age', 'derived_sex', 'derived_race', 'derived_ethnicity', 'loan_type', 'county_code', 'denial_reason-1']


# In[22]:


#Model Training and Testing

#Select Features for models
X = fin_data[['derived_sex', 'derived_race', 'derived_ethnicity', 'county_code']]
y = fin_data[['denial_reason-1']]
#Split the data 80/20
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.20, random_state=42)
#Transform the data to prepare for training
var = ['derived_sex', 'derived_race', 'derived_ethnicity', 'county_code']
var2 = ['denial_reason-1']
X_train = preprocessor.fit_transform(train_x[var])
preprocessor.fit(train_y[var2])
Y_train = preprocessor.transform(train_y[var2])

#Model Testing
#Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, Y_train)
lin_reg.score(X_train, Y_train)

#Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(max_depth = 5)
fit_tree = tree_reg.fit(X_train, Y_train)
#quick accuracy check
from sklearn.model_selection import cross_val_score
print(cross_val_score(fit_tree, X_train, Y_train, cv=5))

#Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
X_train, Y_train = make_regression(n_features=4, n_informative=2,
                       random_state=0, shuffle=False)
regr = RandomForestRegressor(max_depth=2, random_state=0)
fit_regr = regr.fit(X_train, Y_train)
#quick accuracy check
print(cross_val_score(fit_regr, X_train, Y_train, cv=5))


#Gradient Boost Regressor
from sklearn.ensemble import GradientBoostingRegressor
gbrt = GradientBoostingRegressor(max_depth = 5, n_estimators = 3, learning_rate = 1.0)
fit_gbrt = gbrt.fit(X_train, Y_train)
#quick accuracy check
print(cross_val_score(fit_gbrt, X_train, Y_train, cv=3))


#Best Model
#Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
X_train, Y_train = make_regression(n_features=19,
                       random_state=42, shuffle=False)
for_regr = RandomForestRegressor(random_state = 42)
fit_for_regr = for_regr.fit(X_train, Y_train)
print(cross_val_score(fit_regr, X_train, Y_train, cv=5))
#Grid Search to find best parameters
from sklearn.model_selection import GridSearchCV
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [0, 2, 4, 6, 8, 10, 12, 16, 18], 'max_depth': [10, 30, 50, 70, 90, 100, None]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [0, 2, 4, 6, 8, 10, 12, 14, 16, 18], 'max_depth': [10, 30, 50, 70, 90, 100, None]}
]

grid_search = GridSearchCV(for_regr, param_grid, cv = 5, 
                           scoring = 'neg_mean_squared_error', return_train_score = True)

grid_search.fit(X_train, Y_train)
#Determine which parameters produce the highest accuracy
grid_search.best_params_
#Tuned Model
#parameters
max_depth = 10 
max_features = 18
n_estimators = 30
#Final Model
X_train, Y_train = make_regression(n_features=19,
                       random_state=42, shuffle=False)
for_regr = RandomForestRegressor(max_depth = max_depth, max_features = max_features, 
                                 n_estimators = n_estimators,
                                 random_state = 42)
fit_for_regr = for_regr.fit(X_train, Y_train)
print(cross_val_score(fit_regr, X_train, Y_train, cv=5))

#Final Predictions
X_test = preprocessor.fit_transform(test_x[var])
preprocessor.fit(test_y[var2])
Y_test = preprocessor.transform(test_y[var2])
#Model Evaluation
from sklearn.metrics import mean_squared_error
y_pred = for_regr.predict(X_test)
scores = cross_val_score(for_regr, X_test, Y_test, cv=10, scoring='neg_mean_absolute_error')
print(sum(scores) / 10)
#Function to Predict Reason for Denial
def predict_new(sex, race, ethnicity, cty_cd):
    x_data = [sex, race, ethnictiy, ctd_cd]
    mod_data = x_data.preprocessor.fit_transform(x_data)
    pred_y = for_reg(mod_data)
    out = preprocessor.invserse_transform(pred_y)
    print('Denial Reason:' + output)


# In[ ]:




