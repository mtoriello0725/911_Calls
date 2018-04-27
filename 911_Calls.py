# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 13:44:46 2018

@author: mtoriello0725
"""

'''

Python for Data Science 911 Calls

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt     # plt.show() to display plot
import seaborn as sns


call_data = pd.read_csv('911.csv')		# dataframe of 911 calls

### Basic Questions:
print(call_data.info(),'\n\n', call_data.head(), '\n\n')
top_5_zip = call_data['zip'].value_counts().head(5)
top_5_twp = call_data['twp'].value_counts().head(5)
title_unique = call_data['title'].nunique()
print('Zip Code   #Calls\n',top_5_zip, '\n\n', sep='')
print('Township        #Calls\n',top_5_twp,'\n\n', sep='')
print('The Number of Unique titles for 911 Calls is ',title_unique,'\n\n')


### Create New Column Reason: 
call_data['Reason'] = call_data['title'].apply(lambda title: title.split(':')[0])
print('Reason     #Calls\n',call_data['Reason'].value_counts(), sep='')
# Seaborn Plot: 
sns.set(style='darkgrid')
fig = plt.figure()
ax = fig.add_axes([.15,.1,.8,.8])
ax = sns.countplot(x='Reason', data=call_data)
ax.set_title('Reasons for 911 Calls')
ax.set_ylabel('Call Count')


### Convert timeStamp to datetime object and Create columns for Hour, Month, and Day of Week
call_data['timeStamp'] = pd.to_datetime(call_data['timeStamp'])		# convert string to date/time object
call_data['Hour'] = call_data['timeStamp'].apply(lambda hr: hr.hour)		# create hour column
call_data['Month'] = call_data['timeStamp'].apply(lambda mon: mon.month)		# create month column
call_data['Day_of_Week'] = call_data['timeStamp'].apply(lambda day_of_week: day_of_week.dayofweek)		# create day column
call_data['Day_of_Week'] = call_data['Day_of_Week'].map({0:'Mon', 1:'Tue', 2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'})		# assign dayofweek column with recognizable labels

### Day_of_Week Seaborn Plot
fig2 = plt.figure(figsize=(10,5))
ax2 = fig2.add_axes([.1,.1,.8,.8])
ax2 = sns.countplot(x='Day_of_Week', hue='Reason',data=call_data)
ax2.legend(bbox_to_anchor = [1.05, 1])
ax2.set_ylabel('Call Count')
ax2.set_title('911 Calls by Day of the Week')

### Month Seaborn Plot
fig3 = plt.figure(figsize=(10,5))
ax3 = fig3.add_axes([.1,.1,.8,.8])
ax3 = sns.countplot(x='Month', hue='Reason',data=call_data)
ax3.legend(bbox_to_anchor = [1.05, 1])
ax3.set_ylabel('Call Count')
ax3.set_title('911 Calls by Month')


### Plot Linear Model of Monthly Calls
month_tickers = 'Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec'
month_tickers = month_tickers.split()
byMonth = call_data.groupby('Month').count()		# created byMonth grouping for the dataframe
fig4 = plt.figure()
ax4 = fig4.add_axes([.2,.1,.7,.8])
ax4.grid(color='k', linestyle='--', linewidth=1)
ax4.plot(byMonth['twp'], color='b', linewidth=3)		# print count of monthly calls on line plot, (column called in plot can be any column)
ax4.set(xlabel='Month', ylabel='Call Count', title='911 Calls by Month: Linear Model')

### Plot Linear Model of Monthly calls using Seaborns Linear Model plot
lm = sns.lmplot(x='Month',y='twp',data=byMonth.reset_index())		# linear model plot... must reset index of dataframe.plt.show()
fig5 = lm.fig
fig5.suptitle('Linear Model of Monthly Calls', fontsize=18)
lm.set(xlabel='Month', ylabel='Call Count')


### Create new Column called Date
call_data['Date'] = call_data['timeStamp'].apply(lambda d: d.date())		# Created a new column for the date
byDate = call_data.groupby('Date').count()			# Created byDate grouping for the dataframe
byDateMax = byDate[byDate['twp']==byDate.max()['twp']]['twp']
print(byDateMax)
# Plotting

fig6 = plt.figure(figsize=(10,6))
ax6 = fig6.add_axes([.1,.1,.8,.8,])
ax6.grid(color='k', linestyle='--', linewidth=1)
ax6 = byDate['title'].plot(color='b', linewidth=3)
ax6 = byDateMax.plot(marker='o', markersize=10)
#ax6.annotate('Max # Calls', xy=(3,1), xycoords=byDateMax, textcoords=(.8,.8), arrowprops=dict(facecolor='black', 
#	shrink=0.05),horizontalalignment='right', verticalalignment='top')
ax6.set(xlabel='Date', ylabel='Call Count', title='911 Calls by Date: Dec-10-2015 through Aug-21-2016')


### Subplot of each Reason for calls
fig7, ax7 = plt.subplots(3, sharex=True, sharey=True, figsize=(14,8))
fig7.suptitle('# of Calls by Day for each Reason')
ax7[0].plot(call_data[call_data['Reason']=='EMS'].groupby('Date').count()['title'], linewidth=2)
ax7[0].grid(color='k', linestyle='--', linewidth=1)
ax7[0].set_ylabel('EMS Calls')
ax7[1].plot(call_data[call_data['Reason']=='Fire'].groupby('Date').count()['title'], linewidth=2)
ax7[1].grid(color='k', linestyle='--', linewidth=1)
ax7[1].set_ylabel('Fire Calls')
ax7[2].plot(call_data[call_data['Reason']=='Traffic'].groupby('Date').count()['title'], linewidth=2)
ax7[2].grid(color='k', linestyle='--', linewidth=1)
ax7[2].set_ylabel('Traffic Calls')
fig7.subplots_adjust(hspace=.5)


### Heatmaps for Calls made by Hour and Day of Week
byDayofWeekHour = call_data.groupby(by=['Day_of_Week', 'Hour']).count()['twp'].unstack(level=-1)

fig8 = plt.figure(figsize=(10,6))
ax8 = fig8.add_axes([.1,.1,.8,.8,])
ax8 = sns.heatmap(byDayofWeekHour, cmap="coolwarm")
fig8.suptitle('Heatmap: Hour by Day of the Week')

clm = sns.clustermap(byDayofWeekHour, cmap="coolwarm", figsize=(10,6))
fig9 = clm.fig
fig9.suptitle('Clustermap: Hour by Day of the Week')


### Heatmaps for Calls made by Month and Day of Week
byDayofWeekMonth = call_data.groupby(by=['Day_of_Week', 'Month']).count()['twp'].unstack(level=-1)

fig8 = plt.figure(figsize=(10,6))
ax8 = fig8.add_axes([.1,.1,.8,.8,])
ax8 = sns.heatmap(byDayofWeekMonth, cmap="coolwarm")
fig8.suptitle('Heatmap: Month by Day of the Week')

clm = sns.clustermap(byDayofWeekMonth,cmap='coolwarm', figsize=(10,6))
fig9 = clm.fig
fig9.suptitle('Clustermap: Month by Day of the Week')

plt.show()