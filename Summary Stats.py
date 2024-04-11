#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


citations = pd.read_csv('/Users/David/Downloads/finalfinalcitations.csv')
patents = pd.read_csv('/Users/David/Downloads/finalfinalpatents.csv')
financials = pd.read_csv('/Users/David/Downloads/finalfinalfins.csv')


# In[3]:


pd.set_option('display.max_columns', None)


# In[4]:


patents


# In[5]:


citations


# In[6]:


patent_counts = patents.groupby('Ticker')['PatentID'].nunique().reset_index(name='patent_count')


# In[7]:


patent_counts


# In[8]:


total_companies = patent_counts['Ticker'].nunique()
total_companies


# In[9]:


average_patents = patent_counts['patent_count'].mean()
average_patents


# In[10]:


median_patents = patent_counts['patent_count'].median()
median_patents


# In[11]:


max_patents = patent_counts['patent_count'].max()
max_patents


# In[12]:


min_patents = patent_counts['patent_count'].min()
min_patents


# In[13]:


std_patents = patent_counts['patent_count'].std()
std_patents


# In[14]:


max_patents_tickers = patent_counts[patent_counts['patent_count'] == patent_counts['patent_count'].max()]
max_patents_tickers
patent_counts.sort_values(by='patent_count', ascending=False, inplace=True)
patent_counts.head(10)


# In[15]:


min_patents_tickers = patent_counts[patent_counts['patent_count'] == patent_counts['patent_count'].min()]
min_patents_tickers


# In[16]:


citation_counts = citations.groupby('citation_patent_id')['patent_id'].nunique().reset_index(name='citation_count')


# In[17]:


citation_counts


# In[18]:


total_patents_with_citations = citation_counts['citation_patent_id'].nunique()
total_patents_with_citations


# In[19]:


average_citations = citation_counts['citation_count'].mean()
average_citations


# In[20]:


median_citations = citation_counts['citation_count'].median()
median_citations


# In[21]:


max_citations = citation_counts['citation_count'].max()
max_citations


# In[22]:


min_citations = citation_counts['citation_count'].min()
min_citations


# In[23]:


std_citations = citation_counts['citation_count'].std()
std_citations


# In[24]:


max_citations_ids = citation_counts[citation_counts['citation_count'] == citation_counts['citation_count'].max()]
max_citations_ids


# In[25]:


min_citations_ids = citation_counts[citation_counts['citation_count'] == citation_counts['citation_count'].min()]
min_citations_ids


# In[26]:


max_citations_company = patents[patents['PatentID'] == '5892900']
max_citations_company


# In[27]:


patent_counts = patent_counts.sort_values(by='patent_count', ascending=False)
patent_counts


# In[28]:


citation_counts = citation_counts.sort_values(by='citation_count', ascending=False)
citation_counts


# In[29]:


financials


# In[30]:


average_assets = financials['Total Assets'].mean()
average_assets


# In[31]:


median_assets = financials['Total Assets'].median()
median_assets


# In[32]:


max_assets = financials['Total Assets'].max()
max_assets


# In[33]:


min_assets = financials['Total Assets'].min()
min_assets


# In[34]:


std_assets = financials['Total Assets'].std()
std_assets


# In[35]:


financials[financials['Total Assets'] == 3875393000000]


# In[36]:


average_rd = financials['R&D Expense'].mean()
average_rd


# In[37]:


median_rd = financials['R&D Expense'].median()
median_rd


# In[38]:


max_rd = financials['R&D Expense'].max()
max_rd


# In[39]:


min_rd = financials['R&D Expense'].min()
min_rd


# In[40]:


std_rd = financials['R&D Expense'].std()
std_rd


# In[41]:


financials[financials['R&D Expense'] == -515000]


# In[42]:


average_marketval = financials['Market Value'].mean()
average_marketval


# In[43]:


median_marketval = financials['Market Value'].median()
median_marketval


# In[44]:


max_marketval = financials['Market Value'].max()
max_marketval


# In[45]:


min_marketval = financials['Market Value'].min()
min_marketval


# In[46]:


std_marketval = financials['Market Value'].std()
std_marketval


# In[47]:


financials[financials['Ticker'] == '4936C']


# In[48]:


financials


# In[49]:


financials['R&D to Assets Ratio'] = np.where(financials['Total Assets'] == 0, np.nan, financials['R&D Expense'] / financials['Total Assets'])
financials


# In[50]:


average_rtoa = financials['R&D to Assets Ratio'].mean()
average_rtoa


# In[51]:


median_rtoa = financials['R&D to Assets Ratio'].median()
median_rtoa


# In[52]:


min_rtoa = financials['R&D to Assets Ratio'].min()
min_rtoa


# In[53]:


max_rtoa = financials['R&D to Assets Ratio'].max()
max_rtoa


# In[54]:


std_rtoa = financials['R&D to Assets Ratio'].std()
std_rtoa


# In[55]:


patent_counts


# In[56]:


financials


# In[57]:


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


# In[58]:


financials


# In[59]:


average_rds = financials['R&D Stock'].mean()
average_rds


# In[60]:


median_rds = financials['R&D Stock'].median()
median_rds


# In[61]:


min_rds = financials['R&D Stock'].min()
min_rds


# In[62]:


max_rds = financials['R&D Stock'].max()
max_rds


# In[63]:


std_rds = financials['R&D Stock'].std()
std_rds


# In[64]:


detailed_df = pd.read_csv('/Users/David/Downloads/g_patent.tsv', delimiter='\t', header=0)


# In[65]:


detailed_df


# In[66]:


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
patents.drop(columns=['patent_id'], inplace=True)


# In[67]:


patents


# In[ ]:





# In[68]:


patents = pd.read_csv('/Users/David/Downloads/patentswithdates')


# In[69]:


patents


# In[70]:


delta = 0.15  # Depreciation rate

patents['patent_date'] = pd.to_datetime(patents['patent_date'])

patents['year'] = patents['patent_date'].dt.year


# In[ ]:





# In[71]:


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


# In[72]:


patents


# In[73]:


average_patentst = patents['Patent Stock'].mean()
average_patentst


# In[74]:


median_patentst = patents['Patent Stock'].median()
median_patentst


# In[75]:


min_patentst = patents['Patent Stock'].min()
min_patentst


# In[76]:


max_patentst = patents['Patent Stock'].max()
max_patentst


# In[77]:


std_patentst = patents['Patent Stock'].std()
std_patentst


# In[78]:


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


# In[79]:


citations


# In[80]:


financials


# In[81]:


financials['R&D to Assets Ratio'] = np.where(financials['Total Assets'] == 0, np.nan, financials['R&D Stock'] / financials['Total Assets'])
financials


# In[82]:


average_rtoa = financials['R&D to Assets Ratio'].mean()
average_rtoa


# In[83]:


median_rtoa = financials['R&D to Assets Ratio'].median()
median_rtoa


# In[84]:


min_rtoa = financials['R&D to Assets Ratio'].min()
min_rtoa


# In[85]:


max_rtoa = financials['R&D to Assets Ratio'].max()
max_rtoa


# In[86]:


std_rtoa = financials['R&D to Assets Ratio'].std()
std_rtoa


# In[87]:


financials[financials['Ticker'] == '1044B']


# In[88]:


patentsrd = pd.merge(patents, financials[['Ticker', 'Fiscal Year', 'R&D Stock', 'R&D Expense']],
                     left_on=['Ticker', 'year'], right_on=['Ticker', 'Fiscal Year'], how='left')

# Optionally, drop the 'Fiscal Year' column if it's redundant after the merge
patentsrd.drop(columns=['Fiscal Year'], inplace=True)


# In[ ]:





# In[89]:


patentsrd


# In[90]:


patentsrd['R&D Stock'] = patentsrd['R&D Stock'].fillna(0)
patentsrd['R&D Expense'] = patentsrd['R&D Expense'].fillna(0)
patentsrd['Patents/R&D'] = np.where(patentsrd['R&D Stock'] == 0, np.nan, patentsrd['Patent Stock'] / patentsrd['R&D Stock']*1000000)
patentsrd


# In[91]:


average_patrd = patentsrd['Patents/R&D'].mean()
average_patrd


# In[92]:


median_patrd = patentsrd['Patents/R&D'].median()
median_patrd


# In[93]:


min_patrd = patentsrd['Patents/R&D'].min()
min_patrd


# In[94]:


max_patrd = patentsrd['Patents/R&D'].max()
max_patrd


# In[95]:


std_patrd = patentsrd['Patents/R&D'].std()
std_patrd


# In[96]:


patents = pd.read_csv('/Users/David/Downloads/patentswithdates')


# In[97]:


patents['patent_date'] = pd.to_datetime(patents['patent_date'])

patents['year'] = patents['patent_date'].dt.year


# In[98]:


# Ensure that the PatentID column in patents and citation_patent_id in citations are of the same data type
patents['PatentID'] = patents['PatentID'].astype(str)
citations['citation_patent_id'] = citations['citation_patent_id'].astype(str)

# Merge the citations dataframe with the relevant columns of the patents dataframe
citations_with_dates = pd.merge(citations, patents[['PatentID', 'year', 'Ticker']],
                                left_on='citation_patent_id', right_on='PatentID',
                                how='left')

# Drop the duplicate PatentID column if not needed
citations_with_dates.drop(columns=['PatentID'], inplace=True)


# In[99]:


citations_with_dates[citations_with_dates['Ticker'] == 'AAPL']


# In[ ]:





# In[ ]:





# In[100]:


delta = 0.15

# Step 1: Aggregate citations per year for each Ticker
# Assuming each row in 'citations' represents a citation, so we count rows
annual_citations = citations_with_dates.groupby(['Ticker', 'year']).size().reset_index(name='annual_citations')

# Step 2: Calculate citation stock for each Ticker over years
# Initialize a dataframe to store citation stock results
citation_stocks = pd.DataFrame()

# Iterate through each Ticker to calculate its citation stock over years
for ticker in annual_citations['Ticker'].unique():
    ticker_citations = annual_citations[annual_citations['Ticker'] == ticker].sort_values('year')
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


# In[101]:


citation_stocks


# In[102]:


citation_stocks[citation_stocks['citation_stock'] == 184453.54202232824]


# In[103]:


average_citst = citation_stocks['citation_stock'].mean()
average_citst


# In[104]:


median_citst = citation_stocks['citation_stock'].median()
median_citst


# In[105]:


min_citst = citation_stocks['citation_stock'].min()
min_citst


# In[106]:


max_citst = citation_stocks['citation_stock'].max()
max_citst


# In[107]:


std_citst = citation_stocks['citation_stock'].std()
std_citst


# In[108]:


citation_stocks


# In[109]:


patents


# In[114]:


patents_with_citation_stock = pd.merge(patentsrd, citation_stocks[['Ticker', 'year', 'citation_stock']],
                                       left_on=['Ticker', 'year'], right_on=['Ticker', 'year'],
                                       how='left')

# Drop the 'citation_year' column if it's redundant after the merge
patents_with_citation_stock.drop(columns=['year'], inplace=True)


# In[115]:


patents_with_citation_stock


# In[116]:


patents_with_citation_stock['Cit/Pat'] = np.where(patents_with_citation_stock['citation_stock'] == 0, np.nan, patents_with_citation_stock['citation_stock'] / patents_with_citation_stock['Patent Stock'])
patents_with_citation_stock


# In[117]:


average_citpat = patents_with_citation_stock['Cit/Pat'].mean()
average_citpat


# In[118]:


median_citpat = patents_with_citation_stock['Cit/Pat'].median()
median_citpat


# In[119]:


min_citpat = patents_with_citation_stock['Cit/Pat'].min()
min_citpat


# In[120]:


max_citpat = patents_with_citation_stock['Cit/Pat'].max()
max_citpat


# In[121]:


std_citpat = patents_with_citation_stock['Cit/Pat'].std()
std_citpat

