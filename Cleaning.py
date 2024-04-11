#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


patents = pd.read_csv('/Users/David/Downloads/Thesis/g_assignee_not_disambiguated.tsv', delimiter='\t', header=0)
stocks = pd.read_csv('/Users/David/Downloads/Thesis/stocklist.csv')
cleaned = pd.read_csv('/Users/David/Downloads/Thesis/output_matches.csv')
citations = pd.read_csv('/Users/David/Downloads/Thesis/g_us_patent_citation.tsv', delimiter='\t', header=0)


# In[3]:


stocks


# In[4]:


patents = patents.drop('assignee_sequence', axis=1)
patents = patents.drop('rawlocation_id', axis=1)
patents = patents.drop('assignee_id', axis=1)
patents.dropna(subset=['raw_assignee_organization'], inplace=True)
patents = patents[patents['assignee_type'] != 4]
patents = patents[patents['assignee_type'] != 5]
patents = patents[patents['assignee_type'] != 6]
patents = patents[patents['assignee_type'] != 7]
patents = patents[patents['assignee_type'] != 8]
patents = patents[patents['assignee_type'] != 9]
patents = patents[patents['assignee_type'] != 14]
patents = patents[patents['assignee_type'] != 15]
patents = patents[patents['assignee_type'] != 16]
patents = patents[patents['assignee_type'] != 17]
patents = patents[patents['assignee_type'] != 18]
patents = patents[patents['assignee_type'] != 9]
stocks = stocks.drop_duplicates(subset='conml', keep='first')


# In[5]:


threshold = 95
filtered_df = cleaned[cleaned['Score'] >= threshold]
filtered_df.head(55)
filtered_df


# In[8]:


all_tickers = filtered_df[['Ticker']].copy()


# In[10]:


all_tickers = all_tickers.drop_duplicates(subset='Ticker', keep='first')


# In[11]:


all_tickers


# In[12]:


all_tickers['Ticker'].to_csv('tickers.txt', index=False, header=None)


# In[6]:


citations.head(50)


# In[40]:


#sorted_df = citations.sort_values(by='citation_date')


# In[45]:


#sorted_df


# In[11]:


conscit = citations[citations['citation_patent_id'].isin(filtered_df['PatentID'])]


# In[13]:


conscit


# In[14]:


conscit.to_csv('consolidatedcitations', index=False)
filtered_df.to_csv('consolidatedpatents95threshold', index=False)


# In[10]:


patents

