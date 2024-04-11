#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from thefuzz import fuzz
from thefuzz import process
from joblib import Parallel, delayed


# In[2]:


patents = pd.read_csv('/scratch/network/dbotros/thesispythonjob/g_assignee_not_disambiguated.tsv', delimiter='\t', header=0)
stocks = pd.read_csv('/scratch/network/dbotros/thesispythonjob/stocklist.csv')


# In[3]:


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


# In[4]:


def find_all_matches(patent_name, stock_names):
    return process.extractOne(patent_name, stock_names)


# In[6]:


stock_to_ticker_dict = dict(zip(stocks.conml, stocks.tic))


def get_best_match(stocks, patent_id, patent_name, stock_to_ticker_dict):
    
    matches = find_all_matches(patent_name, stocks['conml'])
    print(matches[0])
    
    return (patent_id, patent_name, matches[0], stock_to_ticker_dict[matches[0]], matches[1])


# In[13]:


n_cores = -1


output_match_list = Parallel(n_jobs = n_cores)(delayed(get_best_match)(stocks, patents.patent_id.iloc[i], patents.raw_assignee_organization.iloc[i], stock_to_ticker_dict) \
                                               for i in range(len(patents.raw_assignee_organization)))


# In[14]:


output_match_list


# In[17]:


# Assuming output_match_list is a list of tuples/lists
# Convert it into a DataFrame
output = pd.DataFrame(output_match_list, columns=['PatentID', 'PatentName','MatchName', 'Ticker', 'Score'])

# Specify your desired CSV file path
csv_file_path = '/scratch/network/dbotros/thesispythonjob/output_matches.csv'

# Export DataFrame to CSV
output.to_csv(csv_file_path, index=False)


# In[ ]:


#len(patents.raw_assignee_organization)


# In[34]:


#patents.iloc[0:20,:]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




