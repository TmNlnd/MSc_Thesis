########################
# Descriptive Statistics
########################


# Import the needed library
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

### Data Preparation ###
########################

# Load in the data sets
df_raw = pd.read_csv('DJI_TN.csv', index_col='Date') 
df_raw = pd.read_csv('NASDAQ_TN.csv', index_col='Date') 
df_raw = pd.read_csv('NYSE_TN.csv', index_col='Date') 
df_raw = pd.read_csv('RUSSELL_TN.csv', index_col='Date') 
df_raw = pd.read_csv('SP_TN.csv', index_col='Date') 


# Select correct data ranges and assign to new object
data = df_raw.iloc[:1984]

# Input required values
seq_len = 60                      # Stands for 60 days
moving_average_day = 0
number_of_stocks = 0
predict_day = 1                   # Stands for how many days ahead

# Identify the name of the stock that is the label of the particular dataset
df_name = data['Name'][0]           

# Remove the 'name' column containing the name of the stock that is the label for the dataset
del data['Name']              
      
# Create the target variable consisting of 0 and 1, depending on whether the market goes down or up
target = (data['Close'][predict_day:] / data['Close'][:-predict_day].values).astype(int)

# Slice the data and return all dates involved as input (Meaning minus last row)
data = data[:-predict_day]          
target.index = data.index

# Remove the first 200 days to clean for created technical indicators
data = data[200:] 
data['target'] = target            # Add the target variable to the main dataset
target = data['target']            # Adjust the target dataset to match the entries of the main dataset
del data['target']                 # Remove the target variable from the main dataset


### Missing Values ###
######################

# Identify the missing values
df_missing_values = data.isnull().sum()

# Identify the missing Values
df_missing_values = data.isnull().sum()
df_missing_values.sort_values(ascending=False)[0:14]

# How many rows contain missing values
data.shape[0] - data.dropna().shape[0]

# Fill missing values with the specified method
df_data_clean = data.fillna(0)              

# Now, how many rows contain missing values
df_data_clean.shape[0] - df_data_clean.dropna().shape[0]


### Datasets ###
################

# Construct above steps for each market
df_DJI        = df_data_clean
df_DJI_target = pd.DataFrame(target)

df_NASDAQ        = df_data_clean 
df_NASDAQ_target = pd.DataFrame(target)

df_NYSE        = df_data_clean
df_NYSE_target = pd.DataFrame(target)

df_RUSSELL        = df_data_clean
df_RUSSELL_target = pd.DataFrame(target)

df_SP        = df_data_clean
df_SP_target = pd.DataFrame(target)


### Target distribution ###
###########################

# Inspect the indice direction distribution
df_DJI_target.value_counts()
df_NASDAQ_target.value_counts()
df_NYSE_target.value_counts()
df_RUSSELL_target.value_counts()
df_SP_target.value_counts()

# Assign the right name to each entry
dataset_name = "DJI"
df_DJI_target['dataset_name'] = dataset_name

dataset_name = "NASDAQ"
df_NASDAQ_target['dataset_name'] = dataset_name

dataset_name = "NYSE"
df_NYSE_target['dataset_name'] = dataset_name

dataset_name = "RUSSELL"
df_RUSSELL_target['dataset_name'] = dataset_name

dataset_name = "SP"
df_SP_target['dataset_name'] = dataset_name

# Create a plot displaying the distribution
df_target_distribution = pd.concat([df_DJI_target, 
                                    df_NASDAQ_target, 
                                    df_NYSE_target, df_RUSSELL_target, 
                                    df_SP_target],
                                   ignore_index=True)

g = sns.catplot(x='dataset_name', 
                hue='target',
                kind='count', 
                data=df_target_distribution, 
                ci=False, 
                aspect=1.1,
                legend=False,
                palette=sns.color_palette(['lightcoral', 'mediumturquoise']))

plt.legend(labels=["Down (0)", "Up (1)"], 
           bbox_to_anchor=[0.7,1.1], 
           ncol=2, 
           frameon=True)
plt.xlabel("$Class$ $Distribution$ $of$ $the$ $Target$ $per$ $Stock$ $Market$ $Index$")
plt.ylabel("$Count$")
plt.show()






### Index graph ###
###################

### DJI
df_DJI_graph_10_17 = df_DJI.loc["2009":"2017-11-15"]["Close"]

plt.figure(figsize=(14, 7))
ax = df_DJI_graph_10_17.plot(lw=2)
ax.tick_params(axis='both', which='major', labelsize=11)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel("$Date$", fontsize=15, weight = 'bold')
ax.set_ylabel("$Close$ $Price$ $In$ $US$ $Dollars$", fontsize=15, weight = 'bold')
ax.grid(axis="x")
plt.xlim(xmin=0.0)
plt.xlim(xmax=2017-11-15)
plt.show()


### NASDAQ

df_NASDAQ_graph_10_17 = df_NASDAQ.loc["2009":"2017-11-15"]["Close"]

plt.figure(figsize=(14, 7))
ax = df_NASDAQ_graph_10_17.plot(lw=2, color='coral')
ax.tick_params(axis='both', which='major', labelsize=11)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel("$Date$", fontsize=15, weight = 'bold')
ax.set_ylabel("$Close$ $Price$ $In$ $US$ $Dollars$", fontsize=15, weight = 'bold')
ax.grid(axis="x")
plt.xlim(xmin=0.0)
plt.xlim(xmax=2017-11-15)
plt.show()


### NYSE

df_NYSE_graph_10_17 = df_NYSE.loc["2009":"2017-11-15"]["Close"]

plt.figure(figsize=(14, 7))
ax = df_NYSE_graph_10_17.plot(lw=2, color='hotpink')
ax.tick_params(axis='both', which='major', labelsize=11)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel("$Date$", fontsize=15, weight = 'bold')
ax.set_ylabel("$Close$ $Price$ $In$ $US$ $Dollars$", fontsize=15, weight = 'bold')
ax.grid(axis="x")
plt.xlim(xmin=0.0)
plt.xlim(xmax=2017-11-15)
plt.show()


### RUSSELL

df_RUSSELL_graph_10_17 = df_RUSSELL.loc["2009":"2017-11-15"]["Close"]

plt.figure(figsize=(14, 7))
ax = df_RUSSELL_graph_10_17.plot(lw=2, color='aquamarine')
ax.tick_params(axis='both', which='major', labelsize=11)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel("$Date$", fontsize=15, weight = 'bold')
ax.set_ylabel("$Close$ $Price$ $In$ $US$ $Dollars$", fontsize=15, weight = 'bold')
ax.grid(axis="x")
plt.xlim(xmin=0.0)
plt.xlim(xmax=2017-11-15)
plt.show()


### S&P500

df_SP_graph_10_17 = df_SP.loc["2009":"2017-11-15"]["Close"]

plt.figure(figsize=(14, 7))
ax = df_SP_graph_10_17.plot(lw=2, color='mediumpurple')
ax.tick_params(axis='both', which='major', labelsize=11)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel("$Date$", fontsize=15, weight = 'bold')
ax.set_ylabel("$Close$ $Price$ $In$ $US$ $Dollars$", fontsize=15, weight = 'bold')
ax.grid(axis="x")
plt.xlim(xmin=0.0)
plt.xlim(xmax=2017-11-15)
plt.show()





### Feature Importance ###
##########################

def colors_from_values(values, palette_name):
    normalized = (values - min(values)) / (max(values) - min(values))
    # convert to indices
    indices = np.round(normalized * (len(values) - 1)).astype(np.int32)
    # use the indices to get the colors
    palette = sns.color_palette(palette_name, len(values))
    return np.array(palette).take(indices, axis=0)

fs = SelectKBest(score_func=f_classif)


### DJI
X_selected = fs.fit_transform(df_DJI, df_DJI_target)
y = fs.scores_
x = df_DJI.columns

fig = plt.figure(figsize=(8, 20))
ax = fig.add_subplot(111)
sns.barplot(y, x, ax=ax, palette=colors_from_values(y, "dark:salmon_r"))
plt.xticks(fontsize=13)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xlabel("$Feature$ $Scores$", fontsize=15, weight = 'bold')
plt.ylabel("$92$ $Features$", fontsize=15, weight = 'bold')
plt.title("Analysis of variance correlation coefficient for the DJI")
plt.grid(axis="x")
plt.show()


### NASDAQ
X_selected = fs.fit_transform(df_NASDAQ, df_NASDAQ_target)
y = fs.scores_
x = df_NASDAQ.columns

fig = plt.figure(figsize=(8, 20))
ax = fig.add_subplot(111)
sns.barplot(y, x, ax=ax, palette=colors_from_values(y, "dark:salmon_r"))
plt.xticks(fontsize=13)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xlabel("$Feature$ $Scores$", fontsize=15, weight = 'bold')
plt.ylabel("$92$ $Features$", fontsize=15, weight = 'bold')
plt.title("Analysis of variance correlation coefficient for the NASDAQ")
plt.grid(axis="x")
plt.show()


### NYSE
X_selected = fs.fit_transform(df_NYSE, df_NYSE_target)
y = fs.scores_
x = df_NYSE.columns

fig = plt.figure(figsize=(8, 20))
ax = fig.add_subplot(111)
sns.barplot(y, x, ax=ax, palette=colors_from_values(y, "dark:salmon_r"))
plt.xticks(fontsize=13)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xlabel("$Feature$ $Scores$", fontsize=15, weight = 'bold')
plt.ylabel("$92$ $Features$", fontsize=15, weight = 'bold')
plt.title("Analysis of variance correlation coefficient for the NYSE")
plt.grid(axis="x")
plt.show()


### RUSSELL
X_selected = fs.fit_transform(df_RUSSELL, df_RUSSELL_target)
y = fs.scores_
x = df_RUSSELL.columns

fig = plt.figure(figsize=(8, 20))
ax = fig.add_subplot(111)
sns.barplot(y, x, ax=ax, palette=colors_from_values(y, "dark:salmon_r"))
plt.xticks(fontsize=13)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xlabel("$Feature$ $Scores$", fontsize=15, weight = 'bold')
plt.ylabel("$92$ $Features$", fontsize=15, weight = 'bold')
plt.title("Analysis of variance correlation coefficient for the RUSSELL 2000")
plt.grid(axis="x")
plt.show()

### S&P500
X_selected = fs.fit_transform(df_SP, df_SP_target)
y = fs.scores_
x = df_SP.columns

fig = plt.figure(figsize=(8, 20))
ax = fig.add_subplot(111)
sns.barplot(y, x, ax=ax, palette=colors_from_values(y, "dark:salmon_r"))
plt.xticks(fontsize=13)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xlabel("$Feature$ $Scores$", fontsize=15, weight = 'bold')
plt.ylabel("$92$ $Features$", fontsize=15, weight = 'bold')
plt.title("Analysis of variance correlation coefficient for the S&P500")
plt.grid(axis="x")
plt.show()





### Bar Chart Error Bar Results ###
###################################

## CNN

# Load in the 82 data sets
df_cnn_82     = pd.read_csv('2D-models/new CNN 82 results.csv') 

# Bind the means
mean_DJI     = round((df_cnn_82.iloc[10]['DJI']*100),2)
mean_NASDAQ  = round((df_cnn_82.iloc[10]['NASDAQ']*100),2)
mean_NYSE    = round((df_cnn_82.iloc[10]['NYA']*100),2)
mean_RUSSELL = round((df_cnn_82.iloc[10]['RUT']*100),2)
mean_SP      = round((df_cnn_82.iloc[10]['S&P']*100),2)

# Bind the sds
sd_DJI     = round((df_cnn_82.iloc[12]['DJI']*100),2)
sd_NASDAQ  = round((df_cnn_82.iloc[12]['NASDAQ']*100),2)
sd_NYSE    = round((df_cnn_82.iloc[12]['NYA']*100),2)
sd_RUSSELL = round((df_cnn_82.iloc[12]['RUT']*100),2)
sd_SP      = round((df_cnn_82.iloc[12]['S&P']*100),2)

# Load in the 92 data sets
df_cnn_92     = pd.read_csv('2D-models/new CNN 92 results.csv') 

# Bind the means
mean_DJI_92     = round((df_cnn_92.iloc[10]['DJI']*100),2)
mean_NASDAQ_92  = round((df_cnn_92.iloc[10]['NASDAQ']*100),2)
mean_NYSE_92    = round((df_cnn_92.iloc[10]['NYA']*100),2)
mean_RUSSELL_92 = round((df_cnn_92.iloc[10]['RUT']*100),2)
mean_SP_92      = round((df_cnn_92.iloc[10]['S&P']*100),2)

# Bind the stds
sd_DJI_92     = round((df_cnn_92.iloc[12]['DJI']*100),2)
sd_NASDAQ_92  = round((df_cnn_92.iloc[12]['NASDAQ']*100),2)
sd_NYSE_92    = round((df_cnn_92.iloc[12]['NYA']*100),2)
sd_RUSSELL_92 = round((df_cnn_92.iloc[12]['RUT']*100),2)
sd_SP_92      = round((df_cnn_92.iloc[12]['S&P']*100),2)


# Create the datasets needed
means_82, std_82 = (mean_DJI, mean_NASDAQ, mean_NYSE, mean_RUSSELL, mean_SP), (sd_DJI, sd_NASDAQ, sd_NYSE, sd_RUSSELL, sd_SP)
means_92, std_92 = (mean_DJI_92, mean_NASDAQ_92, mean_NYSE_92, mean_RUSSELL_92, mean_SP_92), (sd_DJI_92, sd_NASDAQ_92, sd_NYSE_92, sd_RUSSELL_92, sd_SP_92)

ind = np.arange(len(means_82))  # the x locations for the groups
width = 0.35  # the width of the bars

# Create the bar chart
fig, ax = plt.subplots(figsize=(9,7))
rects1 = ax.bar(ind - width/2, means_82, width, yerr=std_82,
                label='82 Features')
rects2 = ax.bar(ind + width/2, means_92, width, yerr=std_92,
                label='92 Features')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Macro-F1-Score (%)', fontsize=15)
ax.set_xlabel('Stock Market Indices', fontsize=15)
ax.set_xticks(ind)
ax.set_xticklabels(('DJI', 'NASDAQ', 'NYSE', 'RUSSELL', 'S&P500'), fontsize=13)
ax.set_ylim(top=62)
ax.legend()

def autolabel(rects, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0, 'right': 1, 'left': -1}

    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(offset[xpos]*3, 3),  # use 3 points offset
                    textcoords="offset points",  # in both directions
                    ha=ha[xpos], va='bottom')


autolabel(rects1, "left")
autolabel(rects2, "right")

fig.tight_layout()

plt.show()


### Bar Chart Error Bar Results ###
###################################

## ANN

# Load in the 82 data sets
df_ann_82     = pd.read_csv('2D-models/new ANN 82 results.csv') 

# Bind the means
mean_DJI     = round((df_ann_82.iloc[10]['DJI']*100),2)
mean_NASDAQ  = round((df_ann_82.iloc[10]['NASDAQ']*100),2)
mean_NYSE    = round((df_ann_82.iloc[10]['NYA']*100),2)
mean_RUSSELL = round((df_ann_82.iloc[10]['RUT']*100),2)
mean_SP      = round((df_ann_82.iloc[10]['S&P']*100),2)

# Bind the sds
sd_DJI     = round((df_ann_82.iloc[12]['DJI']*100),2)
sd_NASDAQ  = round((df_ann_82.iloc[12]['NASDAQ']*100),2)
sd_NYSE    = round((df_ann_82.iloc[12]['NYA']*100),2)
sd_RUSSELL = round((df_ann_82.iloc[12]['RUT']*100),2)
sd_SP      = round((df_ann_82.iloc[12]['S&P']*100),2)

# Load in the 92 data sets
df_ann_92     = pd.read_csv('2D-models/new ANN 92 results.csv') 

# Bind the means
mean_DJI_92     = round((df_ann_92.iloc[10]['DJI']*100),2)
mean_NASDAQ_92  = round((df_ann_92.iloc[10]['NASDAQ']*100),2)
mean_NYSE_92    = round((df_ann_92.iloc[10]['NYA']*100),2)
mean_RUSSELL_92 = round((df_ann_92.iloc[10]['RUT']*100),2)
mean_SP_92      = round((df_ann_92.iloc[10]['S&P']*100),2)

# Bind the stds
sd_DJI_92     = round((df_ann_92.iloc[12]['DJI']*100),2)
sd_NASDAQ_92  = round((df_ann_92.iloc[12]['NASDAQ']*100),2)
sd_NYSE_92    = round((df_ann_92.iloc[12]['NYA']*100),2)
sd_RUSSELL_92 = round((df_ann_92.iloc[12]['RUT']*100),2)
sd_SP_92      = round((df_ann_92.iloc[12]['S&P']*100),2)


# Create the datasets needed
means_82, std_82 = (mean_DJI, mean_NASDAQ, mean_NYSE, mean_RUSSELL, mean_SP), (sd_DJI, sd_NASDAQ, sd_NYSE, sd_RUSSELL, sd_SP)
means_92, std_92 = (mean_DJI_92, mean_NASDAQ_92, mean_NYSE_92, mean_RUSSELL_92, mean_SP_92), (sd_DJI_92, sd_NASDAQ_92, sd_NYSE_92, sd_RUSSELL_92, sd_SP_92)

ind = np.arange(len(means_82))  # the x locations for the groups
width = 0.35  # the width of the bars

# Create the bar chart
fig, ax = plt.subplots(figsize=(9,7))
rects1 = ax.bar(ind - width/2, means_82, width, yerr=std_82,
                label='82 Features')
rects2 = ax.bar(ind + width/2, means_92, width, yerr=std_92,
                label='92 Features')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Macro-F1-Score (%)', fontsize=15)
ax.set_xlabel('Stock Market Indices', fontsize=15)
ax.set_xticks(ind)
ax.set_xticklabels(('DJI', 'NASDAQ', 'NYSE', 'RUSSELL', 'S&P500'), fontsize=13)
ax.set_ylim(top=62)
ax.legend()

def autolabel(rects, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0, 'right': 1, 'left': -1}

    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(offset[xpos]*3, 3),  # use 3 points offset
                    textcoords="offset points",  # in both directions
                    ha=ha[xpos], va='bottom')


autolabel(rects1, "left")
autolabel(rects2, "right")

fig.tight_layout()

plt.show()


### Bar Chart Error Bar Results ###
###################################

## SVM

# Load in the 82 data sets
df_svm_82     = pd.read_csv('2D-models/new SVM 82 results.csv') 

# Bind the means
mean_DJI     = round((df_svm_82.iloc[0]['DJI']*100),2)
mean_NASDAQ  = round((df_svm_82.iloc[0]['NASDAQ']*100),2)
mean_NYSE    = round((df_svm_82.iloc[0]['NYA']*100),2)
mean_RUSSELL = round((df_svm_82.iloc[0]['RUT']*100),2)
mean_SP      = round((df_svm_82.iloc[0]['S&P']*100),2)

# Load in the 92 data sets
df_svm_92     = pd.read_csv('2D-models/new SVM 92 results.csv') 

# Bind the means
mean_DJI_92     = round((df_svm_92.iloc[0]['DJI']*100),2)
mean_NASDAQ_92  = round((df_svm_92.iloc[0]['NASDAQ']*100),2)
mean_NYSE_92    = round((df_svm_92.iloc[0]['NYA']*100),2)
mean_RUSSELL_92 = round((df_svm_92.iloc[0]['RUT']*100),2)
mean_SP_92      = round((df_svm_92.iloc[0]['S&P']*100),2)


# Create the datasets needed
means_82, std_82 = (mean_DJI, mean_NASDAQ, mean_NYSE, mean_RUSSELL, mean_SP), (0,0,0,0,0)
means_92, std_92 = (mean_DJI_92, mean_NASDAQ_92, mean_NYSE_92, mean_RUSSELL_92, mean_SP_92), (0,0,0,0,0)

ind = np.arange(len(means_82))  # the x locations for the groups
width = 0.35  # the width of the bars

# Create the bar chart
fig, ax = plt.subplots(figsize=(9,7))
rects1 = ax.bar(ind - width/2, means_82, width, label='82 Features')
rects2 = ax.bar(ind + width/2, means_92, width, label='92 Features')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Macro-F1-Score (%)', fontsize=15)
ax.set_xlabel('Stock Market Indices', fontsize=15)
ax.set_xticks(ind)
ax.set_xticklabels(('DJI', 'NASDAQ', 'NYSE', 'RUSSELL', 'S&P500'), fontsize=13)
ax.set_ylim(top=62)
ax.legend()

def autolabel(rects, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0, 'right': 1, 'left': -1}

    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(offset[xpos]*3, 3),  # use 3 points offset
                    textcoords="offset points",  # in both directions
                    ha=ha[xpos], va='bottom')


autolabel(rects1, "left")
autolabel(rects2, "right")

fig.tight_layout()

plt.show()
