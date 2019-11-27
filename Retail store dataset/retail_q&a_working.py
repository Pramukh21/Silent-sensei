import pandas as pd

retail_set = pd.read_csv('RetailDataSet.csv')

retail_set.head() 

retail_set.fillna(retail_set.mean(),inplace = True)

retail_set.isnull().sum()

retail_set.columns

by_week = retail_set.groupby('WEEK')
by_store_no = retail_set.groupby('STORE_NO')
by_city = retail_set.groupby('CITY')
by_state = retail_set.groupby('STATE')
by_format = retail_set.groupby('FORMAT')
by_region = retail_set.groupby('REGION')
by_special =retail_set.groupby('SPECIAL')

#Which city is the hottest?

print("\nThe hottest cities are :\n",retail_set[retail_set['WEEK_MAX_TEMP']== retail_set['WEEK_MAX_TEMP'].max()]['CITY'])

#Where are we seeing the maximum/ minimum shortages?

print("\nThe places where we can see maximum shortage are : \n\n",retail_set[retail_set['SHORTAGE'] == retail_set['SHORTAGE'].max()]['CITY'])
print("\nThe places where we can see minimum shortage are : \n\n",retail_set[retail_set['SHORTAGE'] == retail_set['SHORTAGE'].min()]['CITY'])

#In terms of average sales, which format is the most profitable?

format1 = by_format['RS_SALES'].mean()
print("\nThe format having highest average profit is :\n",format1[format1 == format1.max()])

#For each format, calculate the store performance?

format2 = by_format['UNIT_SALES'].mean()

print("\n The store performance for each format is:\n",format2)

#In which format were we able to give the lowest average cost (=Sales/Units)?
format3 = by_format['UNIT_SALES'].mean()
print("\nThe format having lowest average cost is :\n",format3[format3 == format3.min()])


#Is Sales Correlated to any of the variables?

retail_set.corr()

#Rs_sales is strongly correlated with unitsales , week_max_temp , week_min_temp , shortage , population
#Rs_sales is weakly correlateed with store number , rain_mm and week
#unit_sales is strongly correlated with rs_sales,week_min_temp,week_max_temp,shortage,population
#unit sales is weakly correlated with week,store_no,rain _mm