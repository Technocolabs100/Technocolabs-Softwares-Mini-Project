# Technocolabs-Softwares-Mini-Project
dir
# #-the problem
# The data scientists at BigMart have collected 2013 sales data for 1559 products across 10 stores in different cities.
# Also, certain attributes of each product and store have been defined. The aim of this data science project is to build a 
# predictive model and find out the sales of each product at a particular store ,
# Using this model, BigMart will try to understand the properties of products and stores which play a key role in increasing sales
# #-hypothesis generation
# alternate hypothesis
#sales increase with low weight ,low fat content,high visibility,recent est.year,outlet size
# null hypothesis
#sales not related to item type,mrp,outlet location,outlet type
# #-loading data and packages
import pandas as pd
import numpy as np
df=pd.read_csv('/content/drive/MyDrive/Train.csv')
df.head()
#data structure and content
#Sorting Pandas Dataframe 
df=pd.read_csv('/content/drive/MyDrive/Train.csv')
 #Add by variable name(s) to sort
newdf = df.sort_values(by='Item_Outlet_Sales')
newdf
#Let’s look at the some of the visualizations to understand below behavior of variable(s) .
# The distribution of item_weight
# Relation between ageitem_weight and sales; and
# If sales are normally distributed or not?
#Remove Duplicate Values based on values of variables "Item_identifier" and "Outlet_identifier"
rem_dup=df.drop_duplicates(['Item_Identifier', 'Outlet_Identifier'])
print(rem_dup)
#To understand the count, average and sum of variable, use dataframe.describe() with Pandas groupby().
test= df.groupby(['Item_Fat_Content'])
test.describe()
test= df.groupby(['Outlet_Size'])
test.describe()
#null
# Identify missing values of dataframe
df.isnull()
#exploratory data analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
%matplotlib inline
df.boxplot(column=['Item_Weight'])
plt.show()
df.shape
df.info()
df.describe()
df.isnull().sum()
#outlet_Size 2410 categorical,itewm_weight 1463 numerical
#we will replace item weight by mean,outlet size by mode
mean1=df['Item_Weight'].mean()
df['Item_Weight'].replace(np.nan,mean1,inplace=True)
df.isnull().sum()
mode1=df