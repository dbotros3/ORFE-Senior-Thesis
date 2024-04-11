#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import curve_fit
from scipy.stats import t


# In[2]:


citations = pd.read_csv('/Users/David/Downloads/finalfinalcitations.csv')
patents = pd.read_csv('/Users/David/Downloads/finalfinalpatents.csv')
financials = pd.read_csv('/Users/David/Downloads/finalfinalfins.csv')


# In[3]:


delta = 0.15  # Depreciation rate

# Ensure dataframe is sorted by 'Ticker' and 'Fiscal Year'
financials.sort_values(by=['Ticker', 'Fiscal Year'], inplace=True)

# Calculate initial R&D stock based on the first available R&D expense for each company
financials['R&D Stock'] = financials.groupby('Ticker')['R&D Expense'].transform(lambda x: x.iloc[0] / delta)

# Use forward fill to handle any intermediate NA values within each company
financials['R&D Stock'] = financials.groupby('Ticker')['R&D Stock'].ffill()

# Set the initial R&D stock for each company's first year
financials.loc[financials.groupby('Ticker').head(1).index, 'R&D Stock'] = financials.loc[financials.groupby('Ticker').head(1).index, 'R&D Expense'] / delta

# Apply the perpetual inventory method for subsequent years
financials['R&D Stock'] = financials.groupby('Ticker').apply(lambda x: x['R&D Expense'] + (1 - delta) * x['R&D Stock'].shift(1)).reset_index(drop=True)

# Fill any remaining NA values in 'R&D Stock' with initial calculation
financials['R&D Stock'].fillna(financials['R&D Expense'] / delta, inplace=True)

# Display the dataframe to confirm the R&D Stock calculation
#financials[['Ticker', 'Fiscal Year', 'R&D Expense', 'R&D Stock']].head()
financials


# In[4]:


financials['R&D to Assets Ratio'] = np.where(financials['Total Assets'] == 0, np.nan, financials['R&D Stock'] / financials['Total Assets'])
financials['Tobin Q'] = np.where(financials['Total Assets'] == 0, np.nan, financials['Market Value'] / financials['Total Assets'])
financials


# In[5]:


financials['R&D to Assets Ratio'].mean()


# In[6]:


#detailed patent info
detailed_df = pd.read_csv('/Users/David/Downloads/g_patent.tsv', delimiter='\t', header=0)


# In[7]:


# adds year to my original patents dataset

patents['PatentID'] = patents['PatentID'].astype(str)
detailed_df['patent_id'] = detailed_df['patent_id'].astype(str)

# Trim whitespace
patents['PatentID'] = patents['PatentID'].str.strip()
detailed_df['patent_id'] = detailed_df['patent_id'].str.strip()

# Ensure case consistency, e.g., convert to upper case
patents['PatentID'] = patents['PatentID'].str.upper()
detailed_df['patent_id'] = detailed_df['patent_id'].str.upper()

# Now, perform the merge operation as before
patents = pd.merge(patents, detailed_df[['patent_id', 'patent_date']], left_on='PatentID', right_on='patent_id', how='left')
#patents.drop(columns=['patent_id'], inplace=True)


# In[8]:


patents


# In[9]:


patentswithdates = pd.read_csv('/Users/David/Downloads/patentswithdates')


# In[10]:


# make a column for just year 
patents['patent_date'] = pd.to_datetime(patents['patent_date'])

patents['year'] = patents['patent_date'].dt.year


# In[11]:


# Ensure dataframe is sorted by 'Ticker' and 'Fiscal Year'
patents.sort_values(by=['Ticker', 'patent_date'], inplace=True)

patents = patents.groupby(['Ticker', 'year']).size().reset_index(name='patent_count')

# Calculate initial R&D stock based on the first available R&D expense for each company
patents['Patent Stock'] = patents.groupby('Ticker')['patent_count'].transform(lambda x: x.iloc[0] / 0.1)

# Use forward fill to handle any intermediate NA values within each company
patents['Patent Stock'] = patents.groupby('Ticker')['Patent Stock'].ffill()

# Set the initial R&D stock for each company's first year
patents.loc[patents.groupby('Ticker').head(1).index, 'Patent Stock'] = patents.loc[patents.groupby('Ticker').head(1).index, 'patent_count'] / delta

# Apply the perpetual inventory method for subsequent years
patents['Patent Stock'] = patents.groupby('Ticker').apply(lambda x: x['patent_count'] + (1 - delta) * x['Patent Stock'].shift(1)).reset_index(drop=True)

# Fill any remaining NA values in 'R&D Stock' with initial calculation
patents['Patent Stock'].fillna(patents['patent_count'] / delta, inplace=True)


# In[12]:


patents


# In[13]:


# put patent stock and r&d stock onto same dataframe, then calculate patent/r&d stock

patentsrd = pd.merge(patents, financials[['Ticker', 'Fiscal Year', 'R&D Stock']],
                     left_on=['Ticker', 'year'], right_on=['Ticker', 'Fiscal Year'], how='left')

# Optionally, drop the 'Fiscal Year' column if it's redundant after the merge
patentsrd.drop(columns=['Fiscal Year'], inplace=True)

patentsrd['R&D Stock'] = patentsrd['R&D Stock'].fillna(0)
patentsrd['Patents/R&D'] = np.where(patentsrd['R&D Stock'] == 0, np.nan, patentsrd['Patent Stock'] / patentsrd['R&D Stock']*1000000)
patentsrd


# In[14]:


# formatting dates for citation dataset

citations['citation_date'] = pd.to_datetime(citations['citation_date'], format='%Y-%m-%d', errors='coerce')

citations['citation_year'] = citations['citation_date'].dt.year

citations['citation_year'] = citations['citation_year'].astype('Int64')


# In[15]:


patentswithdates['patent_date'] = pd.to_datetime(patentswithdates['patent_date'])

patentswithdates['year'] = patentswithdates['patent_date'].dt.year


# In[16]:


# Ensure that the PatentID column in patents and citation_patent_id in citations are of the same data type
patentswithdates['PatentID'] = patentswithdates['PatentID'].astype(str)
citations['citation_patent_id'] = citations['citation_patent_id'].astype(str)

# Merge the citations dataframe with the relevant columns of the patents dataframe
citations_with_dates = pd.merge(citations, patentswithdates[['PatentID', 'year', 'Ticker']],
                                left_on='citation_patent_id', right_on='PatentID',
                                how='left')

# Drop the duplicate PatentID column if not needed
citations_with_dates.drop(columns=['PatentID'], inplace=True)


# In[17]:


# Step 1: Aggregate citations per year for each Ticker
# Assuming each row in 'citations' represents a citation, so we count rows
annual_citations = citations_with_dates.groupby(['Ticker', 'citation_year']).size().reset_index(name='annual_citations')

# Step 2: Calculate citation stock for each Ticker over years
# Initialize a dataframe to store citation stock results
citation_stocks = pd.DataFrame()

# Iterate through each Ticker to calculate its citation stock over years
for ticker in annual_citations['Ticker'].unique():
    ticker_citations = annual_citations[annual_citations['Ticker'] == ticker].sort_values('citation_year')
    ticker_citations['citation_stock'] = 0  # initialize citation stock column
    
    for i in range(len(ticker_citations)):
        if i == 0:
            # For the first year, the citation stock is just the annual citations (no previous stock to depreciate)
            ticker_citations.iloc[i, ticker_citations.columns.get_loc('citation_stock')] = ticker_citations.iloc[i]['annual_citations']
        else:
            # For subsequent years, depreciate last year's stock and add this year's citations
            previous_stock = ticker_citations.iloc[i-1]['citation_stock']
            depreciated_stock = previous_stock * (1 - delta)
            current_year_citations = ticker_citations.iloc[i]['annual_citations']
            ticker_citations.iloc[i, ticker_citations.columns.get_loc('citation_stock')] = depreciated_stock + current_year_citations
    
    # Append this Ticker's citation stocks to the main dataframe
    citation_stocks = pd.concat([citation_stocks, ticker_citations])

# Reset index of the final dataframe
citation_stocks.reset_index(drop=True, inplace=True)


# In[18]:


patentswithdates


# In[19]:


patents_with_citation_stock = pd.merge(patents, citation_stocks[['Ticker', 'citation_year', 'citation_stock']],
                                       left_on=['Ticker', 'year'], right_on=['Ticker', 'citation_year'],
                                       how='left')

# Drop the 'citation_year' column if it's redundant after the merge
#patents_with_citation_stock.drop(columns=['citation_year'], inplace=True)


# In[20]:


patents_with_citation_stock['Cit/Pat'] = np.where(patents_with_citation_stock['citation_stock'] == 0, np.nan, patents_with_citation_stock['citation_stock'] / patents_with_citation_stock['Patent Stock'])
patents_with_citation_stock


# In[21]:


patentsrd


# In[22]:


financials


# In[23]:


patents


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[24]:


# put patent stock and r&d stock onto same dataframe, then calculate patent/r&d stock

patentsrd = pd.merge(patents, financials[['Ticker', 'Fiscal Year', 'R&D Stock', 'R&D to Assets Ratio', 'Tobin Q', 'Total Assets', 'GIC Sector']],
                     left_on=['Ticker', 'year'], right_on=['Ticker', 'Fiscal Year'], how='left')

# Optionally, drop the 'Fiscal Year' column if it's redundant after the merge
patentsrd.drop(columns=['Fiscal Year'], inplace=True)

patentsrd['R&D Stock'] = patentsrd['R&D Stock'].fillna(0)
patentsrd['Patents/R&D'] = np.where(patentsrd['R&D Stock'] == 0, np.nan, patentsrd['Patent Stock'] / patentsrd['R&D Stock']*1000000)
patentsrd


# In[25]:


finalpatents = pd.merge(patentsrd, patents_with_citation_stock[['Ticker', 'year', 'Cit/Pat', 'citation_stock']],
                                       left_on=['Ticker', 'year'], right_on=['Ticker', 'year'],
                                       how='left')


# In[26]:


finalpatents


# In[27]:


finalpatents = finalpatents.replace([np.inf, -np.inf], np.nan).dropna()
finalpatents = finalpatents.dropna()


# In[28]:


finalpatents['Log Tobin Q'] = np.log(finalpatents['Tobin Q'])
finalpatents['log_avg_Tobins_Q'] = finalpatents.groupby('year')['Log Tobin Q'].transform('mean')


# In[ ]:





# In[ ]:





# In[29]:


finalpatents


# In[ ]:





# In[30]:


# OVERALL MODEL WITH NO DUMMIES


# In[31]:


# Define your non-linear function
def model_func(x, gamma1, gamma2, gamma3):
    return np.log(1 + gamma1*x[0] + gamma2*x[1] + gamma3*x[2])

# Prepare your data
xdata = np.vstack((finalpatents['R&D to Assets Ratio'].values, finalpatents['Patents/R&D'].values, finalpatents['Cit/Pat'].values))
ydata = finalpatents['Log Tobin Q'].values 

# Fit the model
params, params_covariance = curve_fit(model_func, xdata, ydata, maxfev=100000)

print(params)  # This will give you the estimated parameters gamma1, gamma2, gamma3


standard_errors = np.sqrt(np.diag(params_covariance))

# Calculate t-statistics for each parameter
t_statistics = params / standard_errors

# Get the degrees of freedom, which is the number of observations minus the number of parameters
degrees_of_freedom = len(ydata) - len(params)

# Assuming a 95% confidence level, calculate the critical t-value
alpha = 0.01  # Significance level for two-tailed test
critical_t = t.ppf(1 - alpha/2, df=degrees_of_freedom)

print("Estimated parameters:", params)
print("Standard errors:", standard_errors)
print("T-statistics:", t_statistics)
print("Critical t-value (95% confidence):", critical_t)

# Determine significance
significant_params = np.abs(t_statistics) > critical_t
print("Significant parameters at 95% confidence level:", significant_params)

p_values = [2 * (1 - t.cdf(np.abs(t_stat), degrees_of_freedom)) for t_stat in t_statistics]

print("P-values:", p_values)



def model_func(x, gamma1, gamma2, gamma3):
    return np.log(1 + gamma1*x[0] + gamma2*x[1] + gamma3*x[2])

# Use the fitted model to predict y values
y_pred = model_func(xdata, *params)

# Calculate the total sum of squares (SST)
y_mean = np.mean(ydata)
SST = np.sum((ydata - y_mean)**2)

# Calculate the residual sum of squares (SSR)
SSR = np.sum((ydata - y_pred)**2)

# Calculate R^2
R_squared = 1 - (SSR / SST)

print(f"R^2: {R_squared}")



# In[ ]:





# In[32]:


# OVERALL MODEL WITH YEAR DUMMIES


# In[33]:


finalpatents['pre_2000'] = (finalpatents['year'] < 2000).astype(int)
finalpatents['post_2000'] = (finalpatents['year'] >= 2000).astype(int)

# Define your non-linear function including the dummy variables
# Note: x[3] is pre_2000, and x[4] is post_2000 in this setup
def model_func(x, gamma1, gamma2, gamma3, delta_pre_2000, delta_post_2000):
    return np.log(1 + gamma1*x[0] + gamma2*x[1] + gamma3*x[2] + delta_pre_2000*x[3] + delta_post_2000*x[4])

# Prepare your data
xdata = np.vstack((finalpatents['R&D to Assets Ratio'].values, 
                   finalpatents['Patents/R&D'].values, 
                   finalpatents['Cit/Pat'].values,
                   finalpatents['pre_2000'].values,
                   finalpatents['post_2000'].values))

ydata = finalpatents['Log Tobin Q'].values

# Fit the model
params, params_covariance = curve_fit(model_func, xdata, ydata, maxfev=100000)

print(params)


standard_errors = np.sqrt(np.diag(params_covariance))

# Calculate t-statistics for each parameter
t_statistics = params / standard_errors

# Get the degrees of freedom, which is the number of observations minus the number of parameters
degrees_of_freedom = len(ydata) - len(params)

# Assuming a 95% confidence level, calculate the critical t-value
alpha = 0.01  # Significance level for two-tailed test
critical_t = t.ppf(1 - alpha/2, df=degrees_of_freedom)

print("Estimated parameters:", params)
print("Standard errors:", standard_errors)
print("T-statistics:", t_statistics)
print("Critical t-value (95% confidence):", critical_t)

# Determine significance
significant_params = np.abs(t_statistics) > critical_t
print("Significant parameters at 95% confidence level:", significant_params)

p_values = [2 * (1 - t.cdf(np.abs(t_stat), degrees_of_freedom)) for t_stat in t_statistics]

print("P-values:", p_values)

y_pred = model_func(xdata, *params)

# Step 2: Calculate SS_res and SS_tot
SS_res = np.sum((ydata - y_pred) ** 2)
SS_tot = np.sum((ydata - np.mean(ydata)) ** 2)

# Step 3: Compute R^2
R_squared = 1 - (SS_res / SS_tot)

R_squared



# In[ ]:





# In[34]:


# FIRST TIME PERIOD MODEL


# In[35]:


filtered_data = finalpatents[(finalpatents['year'] >= 1976) & (finalpatents['year'] < 2000)]

# Define your non-linear function
def model_func(x, gamma1, gamma2, gamma3):
    return np.log(1 + gamma1*x[0] + gamma2*x[1] + gamma3*x[2])

# Prepare your data using the filtered dataset
xdata = np.vstack((filtered_data['R&D to Assets Ratio'].values, filtered_data['Patents/R&D'].values, filtered_data['Cit/Pat'].values))
ydata = filtered_data['Log Tobin Q'].values

# Fit the model
params, params_covariance = curve_fit(model_func, xdata, ydata, maxfev=100000)

print(params)  # This will give you the estimated parameters gamma1, gamma2, gamma3

standard_errors = np.sqrt(np.diag(params_covariance))

# Calculate t-statistics for each parameter
t_statistics = params / standard_errors

# Get the degrees of freedom, which is the number of observations minus the number of parameters
degrees_of_freedom = len(ydata) - len(params)

# Assuming a 95% confidence level, calculate the critical t-value
alpha = 0.01  # Significance level for two-tailed test
critical_t = t.ppf(1 - alpha/2, df=degrees_of_freedom)

print("Estimated parameters:", params)
print("Standard errors:", standard_errors)
print("T-statistics:", t_statistics)
print("Critical t-value (95% confidence):", critical_t)

# Determine significance
significant_params = np.abs(t_statistics) > critical_t
print("Significant parameters at 95% confidence level:", significant_params)

p_values = [2 * (1 - t.cdf(np.abs(t_stat), degrees_of_freedom)) for t_stat in t_statistics]

print("P-values:", p_values)



def model_func(x, gamma1, gamma2, gamma3):
    return np.log(1 + gamma1*x[0] + gamma2*x[1] + gamma3*x[2])

# Use the fitted model to predict y values
y_pred = model_func(xdata, *params)

# Calculate the total sum of squares (SST)
y_mean = np.mean(ydata)
SST = np.sum((ydata - y_mean)**2)

# Calculate the residual sum of squares (SSR)
SSR = np.sum((ydata - y_pred)**2)

# Calculate R^2
R_squared = 1 - (SSR / SST)

print(f"R^2: {R_squared}")


# In[ ]:





# In[36]:


# SECOND PERIOD MODEL


# In[75]:


filtered_data1 = finalpatents[(finalpatents['year'] >= 2000)]

# Define your non-linear function
def model_func(x, gamma1, gamma2, gamma3):
    return np.log(1 + gamma1*x[0] + gamma2*x[1] + gamma3*x[2])

# Prepare your data using the filtered dataset
xdata = np.vstack((filtered_data1['R&D to Assets Ratio'].values, filtered_data1['Patents/R&D'].values, filtered_data1['Cit/Pat'].values))
ydata = filtered_data1['Log Tobin Q'].values

# Fit the model
params, params_covariance = curve_fit(model_func, xdata, ydata, maxfev=100000)

print(params)  # This will give you the estimated parameters gamma1, gamma2, gamma3

standard_errors = np.sqrt(np.diag(params_covariance))

# Calculate t-statistics for each parameter
t_statistics = params / standard_errors

# Get the degrees of freedom, which is the number of observations minus the number of parameters
degrees_of_freedom = len(ydata) - len(params)

# Assuming a 95% confidence level, calculate the critical t-value
alpha = 0.05  # Significance level for two-tailed test
critical_t = t.ppf(1 - alpha/2, df=degrees_of_freedom)

print("Estimated parameters:", params)
print("Standard errors:", standard_errors)
print("T-statistics:", t_statistics)
print("Critical t-value (95% confidence):", critical_t)

# Determine significance
significant_params = np.abs(t_statistics) > critical_t
print("Significant parameters at 95% confidence level:", significant_params)

p_values = [2 * (1 - t.cdf(np.abs(t_stat), degrees_of_freedom)) for t_stat in t_statistics]

print("P-values:", p_values)



def model_func(x, gamma1, gamma2, gamma3):
    return np.log(1 + gamma1*x[0] + gamma2*x[1] + gamma3*x[2])

# Use the fitted model to predict y values
y_pred = model_func(xdata, *params)

# Calculate the total sum of squares (SST)
y_mean = np.mean(ydata)
SST = np.sum((ydata - y_mean)**2)

# Calculate the residual sum of squares (SSR)
SSR = np.sum((ydata - y_pred)**2)

# Calculate R^2
R_squared = 1 - (SSR / SST)

print(f"R^2: {R_squared}")


# In[ ]:





# In[38]:


# OVERALL MODEL WITH GICS DUMMIES


# In[39]:


gic_dummies = pd.get_dummies(finalpatents['GIC Sector'], prefix='GIC')

# Combine the original dataframe with the new dummy variables
finalpatents_expanded = pd.concat([finalpatents, gic_dummies], axis=1)

# Define your non-linear function including the GIC sector dummies
# You need to make sure to account for the correct number of GIC sector dummy variables
def model_func(x, gamma1, gamma2, gamma3, *delta_gic):
    base_model = np.log(1 + gamma1*x[0] + gamma2*x[1] + gamma3*x[2])
    # Add the effect of each GIC sector dummy
    for i in range(len(delta_gic)):
        base_model += delta_gic[i]*x[i+3]
    return base_model

# Prepare your data
# The xdata now needs to include the dummy variables for the GIC sectors
# We need to be careful to align the order of GIC dummies in xdata with the order in the model function
xdata = np.vstack((finalpatents['R&D to Assets Ratio'].values, 
                   finalpatents['Patents/R&D'].values, 
                   finalpatents['Cit/Pat'].values) + 
                   tuple([finalpatents_expanded[col].values for col in gic_dummies.columns]))

ydata = finalpatents['Log Tobin Q'].values 

# Fit the model
# The initial guess for the parameters needs to be extended to include guesses for the delta_gic coefficients
initial_guess = [0.1] * (3 + len(gic_dummies.columns))  # Assuming an initial guess of 0.1 for all parameters

params, params_covariance = curve_fit(model_func, xdata, ydata, p0=initial_guess, maxfev=100000)

print(params)


standard_errors = np.sqrt(np.diag(params_covariance))

# Calculate t-statistics for each parameter
t_statistics = params / standard_errors

# Get the degrees of freedom, which is the number of observations minus the number of parameters
degrees_of_freedom = len(ydata) - len(params)

# Assuming a 95% confidence level, calculate the critical t-value
alpha = 0.05  # Significance level for two-tailed test
critical_t = t.ppf(1 - alpha/2, df=degrees_of_freedom)

print("Estimated parameters:", params)
print("Standard errors:", standard_errors)
print("T-statistics:", t_statistics)
print("Critical t-value (95% confidence):", critical_t)

# Determine significance
significant_params = np.abs(t_statistics) > critical_t
print("Significant parameters at 95% confidence level:", significant_params)

p_values = [2 * (1 - t.cdf(np.abs(t_stat), degrees_of_freedom)) for t_stat in t_statistics]

print("P-values:", p_values)


# In[ ]:





# In[40]:


# GICS SECTOR 10 (ENERGY)


# In[41]:


filtered_data2 = finalpatents[(finalpatents['GIC Sector'] == 10)]

# Define your non-linear function
def model_func(x, gamma1, gamma2, gamma3):
    return np.log(1 + gamma1*x[0] + gamma2*x[1] + gamma3*x[2])

# Prepare your data using the filtered dataset
xdata = np.vstack((filtered_data2['R&D to Assets Ratio'].values, filtered_data2['Patents/R&D'].values, filtered_data2['Cit/Pat'].values))
ydata = filtered_data2['Log Tobin Q'].values

# Fit the model
params, params_covariance = curve_fit(model_func, xdata, ydata, maxfev=100000)

print(params)  # This will give you the estimated parameters gamma1, gamma2, gamma3


standard_errors = np.sqrt(np.diag(params_covariance))

# Calculate t-statistics for each parameter
t_statistics = params / standard_errors

# Get the degrees of freedom, which is the number of observations minus the number of parameters
degrees_of_freedom = len(ydata) - len(params)

# Assuming a 95% confidence level, calculate the critical t-value
alpha = 0.01  # Significance level for two-tailed test
critical_t = t.ppf(1 - alpha/2, df=degrees_of_freedom)

print("Estimated parameters:", params)
print("Standard errors:", standard_errors)
print("T-statistics:", t_statistics)
print("Critical t-value (95% confidence):", critical_t)

# Determine significance
significant_params = np.abs(t_statistics) > critical_t
print("Significant parameters at 95% confidence level:", significant_params)

p_values = [2 * (1 - t.cdf(np.abs(t_stat), degrees_of_freedom)) for t_stat in t_statistics]

print("P-values:", p_values)



def model_func(x, gamma1, gamma2, gamma3):
    return np.log(1 + gamma1*x[0] + gamma2*x[1] + gamma3*x[2])

# Use the fitted model to predict y values
y_pred = model_func(xdata, *params)

# Calculate the total sum of squares (SST)
y_mean = np.mean(ydata)
SST = np.sum((ydata - y_mean)**2)

# Calculate the residual sum of squares (SSR)
SSR = np.sum((ydata - y_pred)**2)

# Calculate R^2
R_squared = 1 - (SSR / SST)

print(f"R^2: {R_squared}")


# In[ ]:





# In[42]:


# GICS SECTOR 15 (MATERIALS)


# In[43]:


filtered_data2 = finalpatents[(finalpatents['GIC Sector'] == 15)]

# Define your non-linear function
def model_func(x, gamma1, gamma2, gamma3):
    return np.log(1 + gamma1*x[0] + gamma2*x[1] + gamma3*x[2])

# Prepare your data using the filtered dataset
xdata = np.vstack((filtered_data2['R&D to Assets Ratio'].values, filtered_data2['Patents/R&D'].values, filtered_data2['Cit/Pat'].values))
ydata = filtered_data2['Log Tobin Q'].values

# Fit the model
params, params_covariance = curve_fit(model_func, xdata, ydata, maxfev=100000)

print(params)  # This will give you the estimated parameters gamma1, gamma2, gamma3


standard_errors = np.sqrt(np.diag(params_covariance))

# Calculate t-statistics for each parameter
t_statistics = params / standard_errors

# Get the degrees of freedom, which is the number of observations minus the number of parameters
degrees_of_freedom = len(ydata) - len(params)

# Assuming a 95% confidence level, calculate the critical t-value
alpha = 0.01  # Significance level for two-tailed test
critical_t = t.ppf(1 - alpha/2, df=degrees_of_freedom)

print("Estimated parameters:", params)
print("Standard errors:", standard_errors)
print("T-statistics:", t_statistics)
print("Critical t-value (95% confidence):", critical_t)

# Determine significance
significant_params = np.abs(t_statistics) > critical_t
print("Significant parameters at 95% confidence level:", significant_params)
p_values = [2 * (1 - t.cdf(np.abs(t_stat), degrees_of_freedom)) for t_stat in t_statistics]

print("P-values:", p_values)



def model_func(x, gamma1, gamma2, gamma3):
    return np.log(1 + gamma1*x[0] + gamma2*x[1] + gamma3*x[2])

# Use the fitted model to predict y values
y_pred = model_func(xdata, *params)

# Calculate the total sum of squares (SST)
y_mean = np.mean(ydata)
SST = np.sum((ydata - y_mean)**2)

# Calculate the residual sum of squares (SSR)
SSR = np.sum((ydata - y_pred)**2)

# Calculate R^2
R_squared = 1 - (SSR / SST)

print(f"R^2: {R_squared}")


# In[ ]:





# In[44]:


# GICS SECTOR 20 (INDUSTRIALS)


# In[45]:


filtered_data2 = finalpatents[(finalpatents['GIC Sector'] == 20)]

# Define your non-linear function
def model_func(x, gamma1, gamma2, gamma3):
    return np.log(1 + gamma1*x[0] + gamma2*x[1] + gamma3*x[2])

# Prepare your data using the filtered dataset
xdata = np.vstack((filtered_data2['R&D to Assets Ratio'].values, filtered_data2['Patents/R&D'].values, filtered_data2['Cit/Pat'].values))
ydata = filtered_data2['Log Tobin Q'].values

# Fit the model
params, params_covariance = curve_fit(model_func, xdata, ydata, maxfev=100000)

print(params)  # This will give you the estimated parameters gamma1, gamma2, gamma3


standard_errors = np.sqrt(np.diag(params_covariance))

# Calculate t-statistics for each parameter
t_statistics = params / standard_errors

# Get the degrees of freedom, which is the number of observations minus the number of parameters
degrees_of_freedom = len(ydata) - len(params)

# Assuming a 95% confidence level, calculate the critical t-value
alpha = 0.01  # Significance level for two-tailed test
critical_t = t.ppf(1 - alpha/2, df=degrees_of_freedom)

print("Estimated parameters:", params)
print("Standard errors:", standard_errors)
print("T-statistics:", t_statistics)
print("Critical t-value (95% confidence):", critical_t)

# Determine significance
significant_params = np.abs(t_statistics) > critical_t
print("Significant parameters at 95% confidence level:", significant_params)
p_values = [2 * (1 - t.cdf(np.abs(t_stat), degrees_of_freedom)) for t_stat in t_statistics]

print("P-values:", p_values)



def model_func(x, gamma1, gamma2, gamma3):
    return np.log(1 + gamma1*x[0] + gamma2*x[1] + gamma3*x[2])

# Use the fitted model to predict y values
y_pred = model_func(xdata, *params)

# Calculate the total sum of squares (SST)
y_mean = np.mean(ydata)
SST = np.sum((ydata - y_mean)**2)

# Calculate the residual sum of squares (SSR)
SSR = np.sum((ydata - y_pred)**2)

# Calculate R^2
R_squared = 1 - (SSR / SST)

print(f"R^2: {R_squared}")


# In[ ]:





# In[46]:


# GICS SECTOR 25 (CONSUMER DISCRETIONARY)


# In[47]:


filtered_data2 = finalpatents[(finalpatents['GIC Sector'] == 25)]

# Define your non-linear function
def model_func(x, gamma1, gamma2, gamma3):
    return np.log(1 + gamma1*x[0] + gamma2*x[1] + gamma3*x[2])

# Prepare your data using the filtered dataset
xdata = np.vstack((filtered_data2['R&D to Assets Ratio'].values, filtered_data2['Patents/R&D'].values, filtered_data2['Cit/Pat'].values))
ydata = filtered_data2['Log Tobin Q'].values

# Fit the model
params, params_covariance = curve_fit(model_func, xdata, ydata, maxfev=100000)

print(params)  # This will give you the estimated parameters gamma1, gamma2, gamma3


standard_errors = np.sqrt(np.diag(params_covariance))

# Calculate t-statistics for each parameter
t_statistics = params / standard_errors

# Get the degrees of freedom, which is the number of observations minus the number of parameters
degrees_of_freedom = len(ydata) - len(params)

# Assuming a 95% confidence level, calculate the critical t-value
alpha = 0.01  # Significance level for two-tailed test
critical_t = t.ppf(1 - alpha/2, df=degrees_of_freedom)

print("Estimated parameters:", params)
print("Standard errors:", standard_errors)
print("T-statistics:", t_statistics)
print("Critical t-value (95% confidence):", critical_t)

# Determine significance
significant_params = np.abs(t_statistics) > critical_t
print("Significant parameters at 95% confidence level:", significant_params)
p_values = [2 * (1 - t.cdf(np.abs(t_stat), degrees_of_freedom)) for t_stat in t_statistics]

print("P-values:", p_values)



def model_func(x, gamma1, gamma2, gamma3):
    return np.log(1 + gamma1*x[0] + gamma2*x[1] + gamma3*x[2])

# Use the fitted model to predict y values
y_pred = model_func(xdata, *params)

# Calculate the total sum of squares (SST)
y_mean = np.mean(ydata)
SST = np.sum((ydata - y_mean)**2)

# Calculate the residual sum of squares (SSR)
SSR = np.sum((ydata - y_pred)**2)

# Calculate R^2
R_squared = 1 - (SSR / SST)

print(f"R^2: {R_squared}")


# In[ ]:





# In[48]:


# GICS SECTOR 30 (CONSUMER STAPLES)


# In[49]:


filtered_data2 = finalpatents[(finalpatents['GIC Sector'] == 30)]

# Define your non-linear function
def model_func(x, gamma1, gamma2, gamma3):
    return np.log(1 + gamma1*x[0] + gamma2*x[1] + gamma3*x[2])

# Prepare your data using the filtered dataset
xdata = np.vstack((filtered_data2['R&D to Assets Ratio'].values, filtered_data2['Patents/R&D'].values, filtered_data2['Cit/Pat'].values))
ydata = filtered_data2['Log Tobin Q'].values

# Fit the model
params, params_covariance = curve_fit(model_func, xdata, ydata, maxfev=100000)

print(params)  # This will give you the estimated parameters gamma1, gamma2, gamma3


standard_errors = np.sqrt(np.diag(params_covariance))

# Calculate t-statistics for each parameter
t_statistics = params / standard_errors

# Get the degrees of freedom, which is the number of observations minus the number of parameters
degrees_of_freedom = len(ydata) - len(params)

# Assuming a 95% confidence level, calculate the critical t-value
alpha = 0.01  # Significance level for two-tailed test
critical_t = t.ppf(1 - alpha/2, df=degrees_of_freedom)

print("Estimated parameters:", params)
print("Standard errors:", standard_errors)
print("T-statistics:", t_statistics)
print("Critical t-value (95% confidence):", critical_t)

# Determine significance
significant_params = np.abs(t_statistics) > critical_t
print("Significant parameters at 95% confidence level:", significant_params)
p_values = [2 * (1 - t.cdf(np.abs(t_stat), degrees_of_freedom)) for t_stat in t_statistics]

print("P-values:", p_values)



def model_func(x, gamma1, gamma2, gamma3):
    return np.log(1 + gamma1*x[0] + gamma2*x[1] + gamma3*x[2])

# Use the fitted model to predict y values
y_pred = model_func(xdata, *params)

# Calculate the total sum of squares (SST)
y_mean = np.mean(ydata)
SST = np.sum((ydata - y_mean)**2)

# Calculate the residual sum of squares (SSR)
SSR = np.sum((ydata - y_pred)**2)

# Calculate R^2
R_squared = 1 - (SSR / SST)

print(f"R^2: {R_squared}")


# In[ ]:





# In[50]:


# GICS SECTOR 35 (HEALTH CARE)


# In[51]:


filtered_data2 = finalpatents[(finalpatents['GIC Sector'] == 35)]

# Define your non-linear function
def model_func(x, gamma1, gamma2, gamma3):
    return np.log(1 + gamma1*x[0] + gamma2*x[1] + gamma3*x[2])

# Prepare your data using the filtered dataset
xdata = np.vstack((filtered_data2['R&D to Assets Ratio'].values, filtered_data2['Patents/R&D'].values, filtered_data2['Cit/Pat'].values))
ydata = filtered_data2['Log Tobin Q'].values

# Fit the model
params, params_covariance = curve_fit(model_func, xdata, ydata, maxfev=100000)

print(params)  # This will give you the estimated parameters gamma1, gamma2, gamma3


standard_errors = np.sqrt(np.diag(params_covariance))

# Calculate t-statistics for each parameter
t_statistics = params / standard_errors

# Get the degrees of freedom, which is the number of observations minus the number of parameters
degrees_of_freedom = len(ydata) - len(params)

# Assuming a 95% confidence level, calculate the critical t-value
alpha = 0.01  # Significance level for two-tailed test
critical_t = t.ppf(1 - alpha/2, df=degrees_of_freedom)

print("Estimated parameters:", params)
print("Standard errors:", standard_errors)
print("T-statistics:", t_statistics)
print("Critical t-value (95% confidence):", critical_t)

# Determine significance
significant_params = np.abs(t_statistics) > critical_t
print("Significant parameters at 95% confidence level:", significant_params)
p_values = [2 * (1 - t.cdf(np.abs(t_stat), degrees_of_freedom)) for t_stat in t_statistics]

print("P-values:", p_values)



def model_func(x, gamma1, gamma2, gamma3):
    return np.log(1 + gamma1*x[0] + gamma2*x[1] + gamma3*x[2])

# Use the fitted model to predict y values
y_pred = model_func(xdata, *params)

# Calculate the total sum of squares (SST)
y_mean = np.mean(ydata)
SST = np.sum((ydata - y_mean)**2)

# Calculate the residual sum of squares (SSR)
SSR = np.sum((ydata - y_pred)**2)

# Calculate R^2
R_squared = 1 - (SSR / SST)

print(f"R^2: {R_squared}")


# In[ ]:





# In[52]:


# GICS SECTOR 40 (FINANCIALS)


# In[53]:


filtered_data2 = finalpatents[(finalpatents['GIC Sector'] == 40)]

# Define your non-linear function
def model_func(x, gamma1, gamma2, gamma3):
    return np.log(1 + gamma1*x[0] + gamma2*x[1] + gamma3*x[2])

# Prepare your data using the filtered dataset
xdata = np.vstack((filtered_data2['R&D to Assets Ratio'].values, filtered_data2['Patents/R&D'].values, filtered_data2['Cit/Pat'].values))
ydata = filtered_data2['Log Tobin Q'].values

# Fit the model
params, params_covariance = curve_fit(model_func, xdata, ydata, maxfev=100000)

print(params)  # This will give you the estimated parameters gamma1, gamma2, gamma3


standard_errors = np.sqrt(np.diag(params_covariance))

# Calculate t-statistics for each parameter
t_statistics = params / standard_errors

# Get the degrees of freedom, which is the number of observations minus the number of parameters
degrees_of_freedom = len(ydata) - len(params)

# Assuming a 95% confidence level, calculate the critical t-value
alpha = 0.01  # Significance level for two-tailed test
critical_t = t.ppf(1 - alpha/2, df=degrees_of_freedom)

print("Estimated parameters:", params)
print("Standard errors:", standard_errors)
print("T-statistics:", t_statistics)
print("Critical t-value (95% confidence):", critical_t)

# Determine significance
significant_params = np.abs(t_statistics) > critical_t
print("Significant parameters at 95% confidence level:", significant_params)
p_values = [2 * (1 - t.cdf(np.abs(t_stat), degrees_of_freedom)) for t_stat in t_statistics]

print("P-values:", p_values)



def model_func(x, gamma1, gamma2, gamma3):
    return np.log(1 + gamma1*x[0] + gamma2*x[1] + gamma3*x[2])

# Use the fitted model to predict y values
y_pred = model_func(xdata, *params)

# Calculate the total sum of squares (SST)
y_mean = np.mean(ydata)
SST = np.sum((ydata - y_mean)**2)

# Calculate the residual sum of squares (SSR)
SSR = np.sum((ydata - y_pred)**2)

# Calculate R^2
R_squared = 1 - (SSR / SST)

print(f"R^2: {R_squared}")


# In[ ]:





# In[54]:


# GICS SECTOR 45 (INFOTECH)


# In[55]:


filtered_data2 = finalpatents[(finalpatents['GIC Sector'] == 45)]

# Define your non-linear function
def model_func(x, gamma1, gamma2, gamma3):
    return np.log(1 + gamma1*x[0] + gamma2*x[1] + gamma3*x[2])

# Prepare your data using the filtered dataset
xdata = np.vstack((filtered_data2['R&D to Assets Ratio'].values, filtered_data2['Patents/R&D'].values, filtered_data2['Cit/Pat'].values))
ydata = filtered_data2['Log Tobin Q'].values

# Fit the model
params, params_covariance = curve_fit(model_func, xdata, ydata, maxfev=100000)

print(params)  # This will give you the estimated parameters gamma1, gamma2, gamma3


standard_errors = np.sqrt(np.diag(params_covariance))

# Calculate t-statistics for each parameter
t_statistics = params / standard_errors

# Get the degrees of freedom, which is the number of observations minus the number of parameters
degrees_of_freedom = len(ydata) - len(params)

# Assuming a 95% confidence level, calculate the critical t-value
alpha = 0.01  # Significance level for two-tailed test
critical_t = t.ppf(1 - alpha/2, df=degrees_of_freedom)

print("Estimated parameters:", params)
print("Standard errors:", standard_errors)
print("T-statistics:", t_statistics)
print("Critical t-value (95% confidence):", critical_t)

# Determine significance
significant_params = np.abs(t_statistics) > critical_t
print("Significant parameters at 95% confidence level:", significant_params)
p_values = [2 * (1 - t.cdf(np.abs(t_stat), degrees_of_freedom)) for t_stat in t_statistics]

print("P-values:", p_values)



def model_func(x, gamma1, gamma2, gamma3):
    return np.log(1 + gamma1*x[0] + gamma2*x[1] + gamma3*x[2])

# Use the fitted model to predict y values
y_pred = model_func(xdata, *params)

# Calculate the total sum of squares (SST)
y_mean = np.mean(ydata)
SST = np.sum((ydata - y_mean)**2)

# Calculate the residual sum of squares (SSR)
SSR = np.sum((ydata - y_pred)**2)

# Calculate R^2
R_squared = 1 - (SSR / SST)

print(f"R^2: {R_squared}")


# In[ ]:





# In[56]:


# GICS SECTOR 50 (COMMS SERVICES)


# In[57]:


filtered_data2 = finalpatents[(finalpatents['GIC Sector'] == 50)]

# Define your non-linear function
def model_func(x, gamma1, gamma2, gamma3):
    return np.log(1 + gamma1*x[0] + gamma2*x[1] + gamma3*x[2])

# Prepare your data using the filtered dataset
xdata = np.vstack((filtered_data2['R&D to Assets Ratio'].values, filtered_data2['Patents/R&D'].values, filtered_data2['Cit/Pat'].values))
ydata = filtered_data2['Log Tobin Q'].values

# Fit the model
params, params_covariance = curve_fit(model_func, xdata, ydata, maxfev=100000)

print(params)  # This will give you the estimated parameters gamma1, gamma2, gamma3


standard_errors = np.sqrt(np.diag(params_covariance))

# Calculate t-statistics for each parameter
t_statistics = params / standard_errors

# Get the degrees of freedom, which is the number of observations minus the number of parameters
degrees_of_freedom = len(ydata) - len(params)

# Assuming a 95% confidence level, calculate the critical t-value
alpha = 0.01  # Significance level for two-tailed test
critical_t = t.ppf(1 - alpha/2, df=degrees_of_freedom)

print("Estimated parameters:", params)
print("Standard errors:", standard_errors)
print("T-statistics:", t_statistics)
print("Critical t-value (95% confidence):", critical_t)

# Determine significance
significant_params = np.abs(t_statistics) > critical_t
print("Significant parameters at 95% confidence level:", significant_params)
p_values = [2 * (1 - t.cdf(np.abs(t_stat), degrees_of_freedom)) for t_stat in t_statistics]

print("P-values:", p_values)



def model_func(x, gamma1, gamma2, gamma3):
    return np.log(1 + gamma1*x[0] + gamma2*x[1] + gamma3*x[2])

# Use the fitted model to predict y values
y_pred = model_func(xdata, *params)

# Calculate the total sum of squares (SST)
y_mean = np.mean(ydata)
SST = np.sum((ydata - y_mean)**2)

# Calculate the residual sum of squares (SSR)
SSR = np.sum((ydata - y_pred)**2)

# Calculate R^2
R_squared = 1 - (SSR / SST)

print(f"R^2: {R_squared}")


# In[ ]:





# In[ ]:





# In[58]:


# GICS SECTOR 55 (UTILITIES)


# In[59]:


filtered_data2 = finalpatents[(finalpatents['GIC Sector'] == 55)]

# Define your non-linear function
def model_func(x, gamma1, gamma2, gamma3):
    return np.log(1 + gamma1*x[0] + gamma2*x[1] + gamma3*x[2])

# Prepare your data using the filtered dataset
xdata = np.vstack((filtered_data2['R&D to Assets Ratio'].values, filtered_data2['Patents/R&D'].values, filtered_data2['Cit/Pat'].values))
ydata = filtered_data2['Log Tobin Q'].values

# Fit the model
params, params_covariance = curve_fit(model_func, xdata, ydata, maxfev=100000)

print(params)  # This will give you the estimated parameters gamma1, gamma2, gamma3


standard_errors = np.sqrt(np.diag(params_covariance))

# Calculate t-statistics for each parameter
t_statistics = params / standard_errors

# Get the degrees of freedom, which is the number of observations minus the number of parameters
degrees_of_freedom = len(ydata) - len(params)

# Assuming a 95% confidence level, calculate the critical t-value
alpha = 0.01  # Significance level for two-tailed test
critical_t = t.ppf(1 - alpha/2, df=degrees_of_freedom)

print("Estimated parameters:", params)
print("Standard errors:", standard_errors)
print("T-statistics:", t_statistics)
print("Critical t-value (95% confidence):", critical_t)

# Determine significance
significant_params = np.abs(t_statistics) > critical_t
print("Significant parameters at 95% confidence level:", significant_params)
p_values = [2 * (1 - t.cdf(np.abs(t_stat), degrees_of_freedom)) for t_stat in t_statistics]

print("P-values:", p_values)



def model_func(x, gamma1, gamma2, gamma3):
    return np.log(1 + gamma1*x[0] + gamma2*x[1] + gamma3*x[2])

# Use the fitted model to predict y values
y_pred = model_func(xdata, *params)

# Calculate the total sum of squares (SST)
y_mean = np.mean(ydata)
SST = np.sum((ydata - y_mean)**2)

# Calculate the residual sum of squares (SSR)
SSR = np.sum((ydata - y_pred)**2)

# Calculate R^2
R_squared = 1 - (SSR / SST)

print(f"R^2: {R_squared}")


# In[ ]:





# In[60]:


# GICS SECTOR 60 (REAL ESTATE)


# In[61]:


filtered_data2 = finalpatents[(finalpatents['GIC Sector'] == 60)]

# Define your non-linear function
def model_func(x, gamma1, gamma2, gamma3):
    return np.log(1 + gamma1*x[0] + gamma2*x[1] + gamma3*x[2])

# Prepare your data using the filtered dataset
xdata = np.vstack((filtered_data2['R&D to Assets Ratio'].values, filtered_data2['Patents/R&D'].values, filtered_data2['Cit/Pat'].values))
ydata = filtered_data2['Log Tobin Q'].values

# Fit the model
params, params_covariance = curve_fit(model_func, xdata, ydata, maxfev=100000)
p_values = [2 * (1 - t.cdf(np.abs(t_stat), degrees_of_freedom)) for t_stat in t_statistics]

print(params)  # This will give you the estimated parameters gamma1, gamma2, gamma3


standard_errors = np.sqrt(np.diag(params_covariance))

# Calculate t-statistics for each parameter
t_statistics = params / standard_errors

# Get the degrees of freedom, which is the number of observations minus the number of parameters
degrees_of_freedom = len(ydata) - len(params)

# Assuming a 95% confidence level, calculate the critical t-value
alpha = 0.01  # Significance level for two-tailed test
critical_t = t.ppf(1 - alpha/2, df=degrees_of_freedom)

print("Estimated parameters:", params)
print("Standard errors:", standard_errors)
print("T-statistics:", t_statistics)
print("Critical t-value (95% confidence):", critical_t)

# Determine significance
significant_params = np.abs(t_statistics) > critical_t
print("Significant parameters at 95% confidence level:", significant_params)

print("P-values:", p_values)



def model_func(x, gamma1, gamma2, gamma3):
    return np.log(1 + gamma1*x[0] + gamma2*x[1] + gamma3*x[2])

# Use the fitted model to predict y values
y_pred = model_func(xdata, *params)

# Calculate the total sum of squares (SST)
y_mean = np.mean(ydata)
SST = np.sum((ydata - y_mean)**2)

# Calculate the residual sum of squares (SSR)
SSR = np.sum((ydata - y_pred)**2)

# Calculate R^2
R_squared = 1 - (SSR / SST)

print(f"R^2: {R_squared}")


# In[ ]:





# In[ ]:





# In[ ]:




