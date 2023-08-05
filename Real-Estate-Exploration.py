import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.stats import norm



# Loading our dataset
df_properties = pd.read_csv("properties_data.csv")
df_properties.head(3)



# Price analysis
df_properties.head()
df_properties.price.describe()

# Looking at above descriptive analysis for property pricing in out daraset, we can say that:
# Property prices ranging from AED 220,000 to 35 Millions
# We have 1906 total properties in our dataset
# Mean price of our apartment is little over 2 Million AED

# cheking for null values
df_properties.isnull().sum().sum()


fig, ax = plt.subplots(figsize = (8, 8))
sns.distplot(df_properties.price);

# Let's check Skewness and Kurtosis and their impact on our analysis, and ultimately prediction models
print("Skewness: %f" % df_properties.price.skew())
print("Kurtosis: %f" % df_properties.price.kurt())


# As we can see that our data is highly skewed and shows a long-tail positive skewness. This uneven distribution is will harm our analysis and prediction model.
# Lets split our dataset in to two parts. We can see from above distribution plot that our tail starts at about 5 Million price point. So lets see how many properties we have under 5 Million and how many above?
df_properties_lt5 = df_properties[df_properties.price < 5000001]
df_properties_gt5 = df_properties[df_properties.price > 5000000]
df_properties_lt5.price.describe()
df_properties_gt5.price.describe()


# In our dataset, we have 1817 properties which are under 5 Million, but only 89 properties are priced at 5 Million to 35 Million.
# As said before, this drastice variance will hurt our model, as we could get a property from the tail section dominating our training dataset with its 35 million price.
# Now, we have option. We either perform a LOG conversion and handle our data, or we can merely eliminate the 89 outliers which are tainting our dataset.
# eliminate 
df_properties = df_properties[df_properties.price < 5000001]



#==============================================================================
#plot the distribution plot again and see the impact
fig, ax = plt.subplots(figsize = (8, 8))
sns.distplot(df_properties.price)

# Let's check Skewness and Kurtosis again and see the impact
print("Skewness: %f" % df_properties.price.skew())
print("Kurtosis: %f" % df_properties.price.kurt())
# Although our skewness is still above 1, it is in much better state than earlier skewness of 6+



#==============================================================================
# Overall Correlation
df_properties_corr = df_properties.copy()
df_properties_corr.drop(['id', 'latitude', 'longitude'], axis=1, inplace=True)
fig, ax = plt.subplots(figsize=(15, 12))
corr_matrix = df_properties_corr.corr()
sns.heatmap(corr_matrix, annot = False)
plt.show()

# Price vs Location
df_neighborhood_price = df_properties[['price', 'neighborhood']].sort_values(by=['price'], ascending = False)
plt.figure(figsize = (30, 10))
plt.bar(df_neighborhood_price.neighborhood, df_neighborhood_price.price, align='center', alpha=0.5)
plt.xticks(rotation='vertical')
plt.show()

# Price per Sqft vs Location
df_neighborhood_price = df_properties[['price_per_sqft', 'neighborhood']].sort_values(by=['price_per_sqft'], ascending = False)

plt.figure(figsize = (30, 10))
plt.bar(df_neighborhood_price.neighborhood, df_neighborhood_price.price_per_sqft, align='center', alpha=0.5)
plt.xticks(rotation='vertical')
plt.show()

# Price vs Area
#scatter plot grlivarea/saleprice
plt.figure(figsize = (10, 8))
feature = 'size_in_sqft'
plt.scatter(df_properties[feature], df_properties['price'])
plt.xlabel('size_in_sqft')
plt.grid(True)


# Price vs Number of Bedrooms
plt.figure(figsize = (10, 8))
plt.scatter(df_properties['size_in_sqft'], df_properties['no_of_bathrooms'])
plt.xlabel('size_in_sqft')
plt.ylabel('no_of_bathrooms')
plt.grid(True)


# We have 1 bedroom apartments ranging upto 5 million, most probably because of the location.
# We have 2 bedrooms apartments ranging from sub 1 million to upto 5 million.
# We have 3 bedrooms apartments ranging from 1 million to up 5 million.

# The location is the most importatn feature, most of the expensive properties are in "Downtown Dubai" and "Palm Jumeirah".
# However, some medium to low quality properties in "Old Town" and "International City" are also affordable.





