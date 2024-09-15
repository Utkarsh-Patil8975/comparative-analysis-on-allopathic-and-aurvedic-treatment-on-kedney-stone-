#!/usr/bin/env python
# coding: utf-8

# ## Practical No 3

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


data=pd.read_csv(r"C:\Users\STATPC19\Downloads\Iris (1).csv")
data


# In[4]:


data.head(10)


# In[5]:


corr=data.corr()
corr


# In[6]:


sns.heatmap(corr,annot=True,cmap='coolwarm')


# ## Pracical No 4

# In[8]:


import numpy as np


# In[9]:


polpulation=np.arange(1,101)
polpulation


# In[1]:


import numpy as np
#create a population of 100 individuals
population=list(range(1,101))
print(population)

#simple random sampling
simple_random_sample=np.random.choice(population,10,replace=False)
print("Simple Random Sample:")
print(simple_random_sample)

#stratified random sampling
strata=np.array_split(population,10)
stratified_sample=[]
for s in strata:
    sample=np.random.choice(s,10,replace=False)
    stratified_sample.extend(sample)
print("Stratified Sample")
print(stratified_sample)

#Systematic sampling
systematic_sample=population[::10]
print("Systeamatic Sample")
print(systematic_sample)

#Cluster sampling
cluster= np.array_split(population,10)
cluster_sample=[]
for c in cluster:
    if np.random.rand()<0.5:
        cluster_sample.extend(c)
print("Cluster Sample")
print(cluster_sample)


# In[1]:


import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# Number of individuals
n = 1000

# Generate random age (18 to 80 years)
age = np.random.randint(18, 81, size=n)

# Generate random gender (0 for female, 1 for male)
gender = np.random.randint(0, 2, size=n)
gender = np.where(gender == 0, 'Female', 'Male')

# Generate random income (20000 to 150000)
income = np.random.randint(20000, 150001, size=n)

# Create DataFrame
data = pd.DataFrame({'Age': age, 'Gender': gender, 'Income': income})

# Display first few rows of the dataset
print(data.head())
# Convenience Sampling (Randomly select 100 individuals)
convenience_sample = data.sample(n=100, random_state=42)

# Display first few rows of convenience sample
# Convenience Sampling (Randomly select 100 individuals)
convenience_sample = data.sample(n=100, random_state=42)
print("\nConvenience Sampling:")
print(convenience_sample.head())


# Display first few rows of purposive sample
# Purposive Sampling (Select individuals with income > 100000)
purposive_sample = data[data['Income'] > 100000].sample(n=100, random_state=42)

# Display first few rows of purposive sample
print("\nPurposive Sampling:")
print(purposive_sample.head())
# Display first few rows of quota sample
# Quota Sampling (Select 50 male and 50 female)
quota_sample_male = data[data['Gender'] == 'Male'].sample(n=50, random_state=42)
quota_sample_female = data[data['Gender'] == 'Female'].sample(n=50, random_state=42)
quota_sample = pd.concat([quota_sample_male, quota_sample_female])

# Display first few rows of quota sample
print("\nQuota Sampling:")
print(quota_sample.head())


## Snowball Sampling (Select 5 random individuals to start snowball)
initial_sample = data.sample(n=5, random_state=42)

# Add new samples based on referrals (randomly select 2 individuals referred by each initial participant)
snowball_sample = initial_sample.copy()
for index, row in initial_sample.iterrows():
    referrals = data.sample(n=2, random_state=index)
    snowball_sample = pd.concat([snowball_sample, referrals])

# Display first few rows of snowball sample
print("\nSnowball Sampling:")
print(snowball_sample.head())


# In[ ]:




