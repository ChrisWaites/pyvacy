#!/usr/bin/python3

################################################################################
# This script generates ground truth data out of the pre-processed dataset and
# its meta-inromation object.

import csv
import json
from random import randint
from time import time
import sys

# Number of buckets for counting of numeric values (integers and floats)
NUM_BUCKETS = 100

# The code will print log message to console when processing each n-th row
# specified here.
NUM_ROWS_BETWEEN_LOG_MESSAGES = 50000

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
inData = csv.reader(open(inDataFilename, newline=''), dialect='excel')
inDataSpecs = json.load(open(inDataSpecsFilename))
dataHeader = next(inData)

# Randomly selects up to 100 sets of 3 columns each.
columnSets = [];
MAX_COLUMN_ID = len(dataHeader) - 1;
while len(columnSets) < 100:
  col3 = randint(0, MAX_COLUMN_ID - 1);
  col2 = randint(0, MAX_COLUMN_ID - 1);
  col1 = randint(0, MAX_COLUMN_ID - 1);
  if (col3 != col2 and col3 != col1 and col2 != col1):
    columnSets.append([col1, col2, col3])

################################################################################
# Collecting stats.

print('Collecting stats...')

counts = {}

def countRow(row, bucket, cols, depth):
  value = row[cols[depth]]
  d = inDataSpecs[dataHeader[cols[depth]]]
  if d['type'] == 'enum': bucketId = int(value)
  elif value == '': bucketId = ''
  else: # integer or float
    bucketSize = 1.0001 * float(d['max'] - d['min']) / NUM_BUCKETS
    bucketId = int((float(value) - d['min']) / bucketSize)
  
  if depth == 2: # We are at leaf > just increment the counter.
    if bucketId in bucket: bucket[bucketId] += 1
    else: bucket[bucketId] = 1
  else:
    if bucketId not in bucket: bucket[bucketId] = {}
    countRow(row, bucket[bucketId], cols, 1 + depth)

rowIndex = 0
for row in inData:
  # Logging
  rowIndex += 1
  if rowIndex % NUM_ROWS_BETWEEN_LOG_MESSAGES == 0:
    print(f'{rowIndex / 1000 :12}k rows processed')
  # Count the row for each ground truth chunk.
  for index in range(len(columnSets)):
    if index not in counts: counts[index] = {}
    countRow(row, counts[index], columnSets[index], 0)

################################################################################
# Output.

print('Output...')

outGT = csv.writer(open(outGroundTruth, 'w', newline=''), dialect='excel')

for testId in counts:
  test = counts[testId]
  outGT.writerow(['@NEWCASE!'] + columnSets[testId])
  for key1 in test:
    for key2 in test[key1]:
      for key3 in test[key1][key2]:
        value = float(test[key1][key2][key3]) / rowIndex
        outGT.writerow([key1, key2, key3, value])

print(f'Time spent {(time() - startedAt) / 60} min')
