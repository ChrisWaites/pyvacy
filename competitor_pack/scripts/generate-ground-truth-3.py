#!/usr/bin/python3

################################################################################
# This script generates ground truth data out of the pre-processed dataset and
# its meta-inromation object.

import csv
import json
from random import randint, uniform
from time import time
import sys

# The code will print log message to console when processing each n-th row
# specified here.
NUM_ROWS_BETWEEN_LOG_MESSAGES = 100000

startedAt = time()

################################################################################
# Initialization.

print('Initialization...')

# Reads command-line arguments:
# - Input dataset;
# - Dictionary (meta-information) on the input dataset;
# - Ground truth;
_, inDataFilename, inDataSpecsFilename, outGroundTruth = sys.argv

# Opens necessary files.
inDataSpecs = json.load(open(inDataSpecsFilename))
inData = csv.reader(open(inDataFilename, newline=''), dialect='excel')
dataHeader = next(inData)

data = {}

SEX_INDEX = dataHeader.index('SEX')
CITY_INDEX = dataHeader.index('CITY')
WAGE_INDEX = dataHeader.index('INCWAGE')

# Collecting data
print('Loading data...')
rowIndex = 0
for row in inData:
  rowIndex += 1
  if rowIndex % NUM_ROWS_BETWEEN_LOG_MESSAGES == 0:
    print(f'{rowIndex / 1000 :12}k rows processed')

  sex = int(row[SEX_INDEX]) - 1
  city = row[CITY_INDEX]
  wage = int(row[WAGE_INDEX])
  if city not in data:
    data[city] = []
  data[city].append([sex, wage])

print('Calculating Gini indices...')

def sortSecond(val):
  return val[1]

results = {}

for city in data:
  d = data[city]
  d.sort(key = sortSecond)
  totalPeople = 0
  totalWage = 0
  for item in d:
    if (item[1] < 999998): # 999998 is N/A code
      totalWage += item[1]
      totalPeople += 1
  g = 0
  accWage = 0
  for item in d:
    if (item[1] < 999998):
      accWage += item[1]
      g += accWage / totalPeople / totalWage
  results[city] = { 'gini': 1 - 2 * g }

paygap = {}
for city in data:
  d = data[city]
  buf = [[0, 0], [0, 0]]
  for item in d:
    if (item[1] < 999998): # 999998 is N/A code
      buf[item[0]][0] += 1
      buf[item[0]][1] += item[1]
  results[city]['paygap'] = buf[0][1] / buf[0][0] - buf[1][1] / buf[1][0]

print('Output...')

outGT = csv.writer(open(outGroundTruth, 'w', newline=''), dialect='excel')

for city in results:
  outGT.writerow([city, results[city]['gini'], results[city]['paygap']])
