import numpy as np
import pandas as pd
from collections import Counter

iris = pd.read_csv('iris.csv')

# 1. How many different names of flowers are in this data set?

print 'Count of unique flower names: %d' %(iris.Name.nunique())

# 2. How many exist of each type of flower?

print iris.groupby('Name').Name.count()

# 3. Determine the min, median, mean, max for each numeric feature in the data set.

print iris.describe()

# 4. Determine the same for each flower type.

print iris.groupby('Name').describe()

# 5a. How does the shape of these results change from the average of all flowers?

print iris.describe().shape, iris.groupby('Name').describe().shape # triple the rows, since there are three names

# 6b. Which features seem to be the most important in determining what kind of flower it is?

# - All setosa flowers have a petal length and petal width which is smaller than the smallest versicolor or virginica.

# Sort the data frame by each column (aside from Name), and print the results of each. What interesting trends exist in this data set, based on distributions?

print iris.sort('SepalLength')
print iris.sort('SepalWidth')
print iris.sort('PetalLength')
print iris.sort('PetalWidth')

#  - In addition to the petal width/length observation above, it appears that versicolor genereally has a shorter sepal length than virginica, though not exclusively.

# **8**. write several functions to apply to the data frame that attempt to organize the data set using strings instead of floats, like our short_or_long function from lecture. Use this to best summarize your data.

def classify_length(record, series):
	if record <= np.percentile(series, 25):
		return 'small'
	elif record >= np.percentile(series, 75):
		return 'large'
	return 'medium'

for column in iris.columns:
	if column != 'Name':
		iris[column] = iris[column].apply(lambda record: classify_length(record, iris[column]))

# **9**. CHALLENGE QUESTION: From everything above, you should be able to predict relatively accurately for each row what kind of flower it is (without using the Name column as an obvious hint). Write a function that uses the data with if and else statements to attempt to classify each row.

def predict_flower_name(record):
	if Counter(record.values)['small'] > 2:
		return 'Iris-setosa'
	if Counter(record.values)['medium'] > 2:
		return 'Iris-versicolor'
	else:
		return 'Iris-virginica'

iris['PredName'] = iris.apply(lambda record: predict_flower_name(record), axis=1)

print iris

print 'Model accuracy: {0:.0%}'.format(1. * len(iris[iris['Name'] == iris['PredName']]) / len(iris))





