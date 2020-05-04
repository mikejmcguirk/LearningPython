#Intakes a set of player box scores from Basketball-Reference.com,
#then checks for a correlation between player scoring and team victory

import numpy as np
from scipy.stats import pearsonr

import pandas as pd
import matplotlib.pyplot as plt

teamResults = pd.read_csv('PlayerResults.csv')

results = np.array(teamResults['Unnamed: 7'])
cleanResults = np.zeros_like(results, dtype = int)

for i in range(len(results)):
    leftCount = results[i].count('(') == 1
    rightCount = results[i].count(')') == 1
    startIndex = int(results[i].find('('))
    stopIndex = int(results[i].find(')'))

    if (leftCount and rightCount and startIndex < stopIndex):    
        cleanResults[i] = int(results[i][startIndex + 1:stopIndex])    
    else:
        cleanResults[i] = 0
        
points = np.array(teamResults['PTS'])

for i in range(len(points)):
    if(not points[i].isnumeric()):
        points[i] = -1

cleanPoints = points.astype(int)

presentMask = cleanPoints >= 0

correlators = pd.DataFrame({'PlayerPoints': cleanPoints[presentMask], \
    'TeamResults': cleanResults[presentMask]})
    
coefficient, _ = \
    pearsonr(correlators['PlayerPoints'], correlators['TeamResults'])
    
#print(coefficient)

plt.scatter(correlators['PlayerPoints'], correlators['TeamResults'], \
    marker = 'o')
plt.xlabel(correlators.columns[0])
plt.ylabel(correlators.columns[1])

z = np.polyfit(correlators['PlayerPoints'], correlators['TeamResults'], 1)
p = np.poly1d(z)
plt.plot(correlators['PlayerPoints'],p(correlators['PlayerPoints']),"r--")

plt.show()