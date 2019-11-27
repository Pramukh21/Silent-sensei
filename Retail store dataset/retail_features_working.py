#Importing the packages
import pandas as pd
import matplotlib.pyplot as plt

#Read the CSV file
retail_set = pd.read_csv("RetailDataSet.csv")

retail_set.fillna(retail_set.mean(),inplace = True)

#To check the number of NAN values in each column
retail_set.isnull().sum()

#To see the first 5 rows of the data
retail_set.head()

#To check what all columns are present
retail_set.columns

#To get the mean of each coloumn
retail_set_mean = retail_set.mean()
#To get the co-relation of each coloumn
retail_set_corr = retail_set.corr()
#To get the variance of each coloumn
retail_Set_var = retail_set.var()
#To get the Standard Deviation of each coloumn
retail_set_std = retail_set.std()

#Grouping by unique values of each columns
by_week = retail_set.groupby('WEEK')
by_store_no = retail_set.groupby('STORE_NO')
by_city = retail_set.groupby('CITY')
by_state = retail_set.groupby('STATE')
by_format = retail_set.groupby('FORMAT')
by_region = retail_set.groupby('REGION')
by_special =retail_set.groupby('SPECIAL')

#To find the number of unique values and also get the unique values of each columns
unique_cities = retail_set['CITY'].unique()
number_of_unique_cities = retail_set['CITY'].nunique()

unique_states = retail_set['STATE'].unique()
number_of_unique_states = retail_set['STATE'].nunique()

unique_format = retail_set['FORMAT'].unique()
number_of_unique_format = retail_set['FORMAT'].nunique()

unique_region = retail_set['REGION'].unique()
number_of_region = retail_set['REGION'].nunique()

unique_special = retail_set['SPECIAL'].unique()
number_of_unique_special = retail_set['SPECIAL'].nunique()

unique_store_no = retail_set['STORE_NO'].unique()
number_of_unique_store_number = retail_set['STORE_NO'].nunique()

#To get the mean of each individual value of a particular column
by_week_mean = by_week.mean()
by_store_no_mean = by_store_no.mean()
by_city_mean = by_city.mean()
by_state_mean = by_state.mean()
by_format_mean = by_format.mean()
by_region_mean = by_region.mean()
by_special_mean = by_special.mean()

#To get the variance of each individual value of a particular column
by_week_var = by_week.var()
by_store_no_var = by_store_no.var()
by_city_var = by_city.var()
by_state_var = by_state.var()
by_format_var = by_format.var()
by_region_var = by_region.var()
by_special_var = by_special.var()

#To get the maximum value of each individual value of a particular column
by_week_max = by_week.max()
by_store_no_max = by_store_no.max()
by_city_max = by_city.max()
by_state_max = by_state.max()
by_format_max = by_format.max()
by_region_max = by_region.max()
by_special_max = by_special.max()

#To get the minimum value of each individual value of a particular column
by_week_min = by_week.min()
by_store_no_min = by_store_no.min()
by_city_min = by_city.min()
by_state_min = by_state.min()
by_format_min = by_format.min()
by_region_min = by_region.min()
by_special_min = by_special.min()

#To get the Standard deviation of each individual value of a particular column
by_week_std = by_week.std()
by_store_no_std = by_store_no.std()
by_city_std = by_city.std()
by_state_std = by_state.std()
by_format_std = by_format.std()
by_region_std = by_region.std()
by_special_std = by_special.std()


#Plotting the graphs for each columns vs RS_SALES column

fig1 = plt.figure()
ax1 = fig1.add_axes([0.1,0.1,1.2,1.2])
ax1.scatter(unique_cities,by_city_mean['RS_SALES'])
plt.xlabel('CITY')
plt.ylabel('RSVALUES')
fig2 = plt.figure()
ax2 = fig2.add_axes([0.1,0.1,1.2,1.2])
ax2.plot(unique_cities,by_city_mean['RS_SALES'])
plt.xlabel('CITY')
plt.ylabel('RSVALUES')


fig3 = plt.figure()

ax3 = fig3.add_axes([0.1,0.1,1.2,1.2])
ax3.scatter(unique_states,by_state_mean['RS_SALES'])
plt.xlabel('STATES')
plt.ylabel('RSVALUES')
fig4 = plt.figure()
ax4 = fig4.add_axes([0.1,0.1,1.6,1.6])
ax4.plot(unique_states,by_state_mean['RS_SALES'])
plt.xlabel('STATES')
plt.ylabel('RSVALUES')


fig5 = plt.figure()
ax5 = fig5.add_axes([0.1,0.1,1.2,1.2])
ax5.scatter(unique_format,by_format_mean['RS_SALES'])
plt.xlabel('FORMAT')
plt.ylabel('RSVALUES')
fig6 = plt.figure()
ax6 = fig6.add_axes([0.1,0.1,1.2,1.2])
ax6.plot(unique_format,by_format_mean['RS_SALES'])
plt.xlabel('FORMAT')
plt.ylabel('RSVALUES')


fig7 = plt.figure()
ax7 = fig7.add_axes([0.1,0.1,1.2,1.2])
ax7.scatter(unique_region,by_region_mean['RS_SALES'])
plt.xlabel('REGION')
plt.ylabel('RSVALUES')
fig8 = plt.figure()
ax8 = fig8.add_axes([0.1,0.1,1.2,1.2])
ax8.plot(unique_region,by_region_mean['RS_SALES'])
plt.xlabel('REGION')
plt.ylabel('RSVALUES')


fig9 = plt.figure()
ax9 = fig9.add_axes([0.1,0.1,1.2,1.2])
ax9.scatter(unique_special,by_special_mean['RS_SALES'])
plt.xlabel('SPECIAL')
plt.ylabel('RSVALUES')
fig10 = plt.figure()
ax10 = fig10.add_axes([0.1,0.1,1.2,1.2])
ax10.plot(unique_special,by_special_mean['RS_SALES'])
plt.xlabel('SPECIAL')
plt.ylabel('RSVALUES')

fig11 = plt.figure()
ax11 = fig11.add_axes([0.1,0.1,1.2,1.2])
ax11.scatter(unique_store_no,by_store_no_mean['RS_SALES'])
plt.xlabel('STORE_NO')
plt.ylabel('RSVALUES')
fig12 = plt.figure()
ax12 = fig12.add_axes([0.1,0.1,1.2,1.2])
ax12.plot(unique_store_no,by_store_no_mean['RS_SALES'])
plt.xlabel('STORE_NO')
plt.ylabel('RSVALUES')

fig13 = plt.figure()
ax13 = fig13.add_axes([0.1,0.1,1.2,1.2])
ax13.scatter(retail_set['UNIT_SALES'],retail_set['RS_SALES'])
plt.xlabel('UNIT SALES')
plt.ylabel('RS_SALES')

fig14 = plt.figure()
ax14 = fig14.add_axes([0.1,0.1,1.2,1.2])
ax14.scatter(retail_set['POPULATION'],retail_set['RS_SALES'])
plt.xlabel('POPULATION')
plt.ylabel('RS_SALES')

print(by_city['WEEK_MAX_TEMP'].max())
print("The maximum shortage is = ",retail_set['SHORTAGE'].max())
print("The minimum shortage is = ",retail_set['SHORTAGE'].min())
