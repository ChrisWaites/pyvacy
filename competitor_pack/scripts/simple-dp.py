#!/usr/bin/python3

################################################################################
# Example of a simple differential data privacy algorithm.
# It intentionally misses any possible performance optimizations.

import csv
import json
import numpy
import sys
from random import uniform

# Number of buckets for counting of numerical values (integer or floats).
NUM_BUCKETS = 100

# The code will print log message to console when processing each n-th row
# specified here.
NUM_ROWS_BETWEEN_LOG_MESSAGES = 250000

################################################################################
# Initialization.

print('Initialization...')

# Reads arguments from the command line (the first value is ignored, as it holds
# executable path):
# - dataFilename - Path to the CSV dataset to privatize;
# - dataSpecsFilename - Path to JSON file, holding meta-information about data;
# - epsilon - Epsilon parameter;
# - maxNumColumns - The maximum number of columns from the input dataset to
#   process. As the implemented simple algorithm is not able to handle large
#   data domain, this option allows to run it on a sub-set of input dataset.

_, inDataFilename, outDataFilename, dataSpecsFilename, epsilon, maxNumColumns = sys.argv
epsilon = float(epsilon)
maxNumColumns = int(maxNumColumns)

# Loads data.

data = csv.reader(open(inDataFilename, newline=''), dialect='excel')
specs = json.load(open(dataSpecsFilename))

# Inits datastructure to count different cominations of values in the input
# dataset.

dataHeader = next(data) # Names of CSV columns

# Recursive step of the initialization
def init(columnId, depth):
  # Interupts recursion when the last column of the dataset is reached,
  # or the we hit the maxNumColumns restriction.
  if columnId == len(dataHeader) or depth == maxNumColumns: return 0

  # Gets meta-data of the current column.
  d = specs[dataHeader[columnId]]

  # Ignores columns containing UUID of data records (moves to the next column).
  if 'uuid' in d and d['uuid']: return init(1 + columnId, depth)

  res = {}

  # For columns with categorical data we need separate buckets for each possible
  # values, thus we init it this way.
  if d['type'] == 'enum':
    for i in range(1 + d['maxval']):
      res[i] = init(1 + columnId, 1 + depth)

  # For other columns (with integer or float number data), we gonna gonna have
  # NUM_BUCKETS buckets:
  else:
    for i in range(NUM_BUCKETS):
      res[i] = init(1 + columnId, 1 + depth)

  return res

counts = init(0, 0)

################################################################################
# Collecting dataset statistics.

print('Collecting dataset statistics...')

# Counts a single row. Recursion is used to locate target bucket.
def countRow(row, buckets, columnId):
  d = specs[dataHeader[columnId]]

  # Ignores columns containing UUID.
  if 'uuid' in d and d['uuid']: return countRow(row, buckets, 1 + columnId)

  # Figures out bucket ID.
  if d['type'] == 'enum': bucketId = int(row[columnId])
  else: # integer or float column
    bucketSize = float(d['max'] - d['min']) / NUM_BUCKETS
    bucketId = int((float(row[columnId]) - d['min']) / bucketSize)
    if bucketId == NUM_BUCKETS: bucketId = NUM_BUCKETS - 1

  # Leaf bucket is reached - increment the count.
  if type(buckets[bucketId]) is int: buckets[bucketId] += 1

  # Otherwise - go into the found bucket.
  else: countRow(row, buckets[bucketId], 1 + columnId)

numRows = 0
for row in data:
  numRows += 1
  if numRows % NUM_ROWS_BETWEEN_LOG_MESSAGES == 0:
    print(f'{numRows / 1000 :12}k rows processed')
  countRow(row, counts, 0)

################################################################################
# Adding Laplase noise to the counts.

print('Adding noise to the counts...')

numOutRows = 0

def addNoise(buckets):
  global numOutRows

  for key in buckets:
    if type(buckets[key]) is int:
      noise = numpy.random.laplace(0, 1.0 / epsilon)
      buckets[key] = round(buckets[key] + noise)

      # If result is negative, we should treat is as zero count.
      if buckets[key] < 0: buckets[key] = 0

      numOutRows += buckets[key]

    else: addNoise(buckets[key])

addNoise(counts);

################################################################################
# Generates output dataset.

print('Generating output...')

outFile = open(outDataFilename, 'w', newline='')
outData = csv.writer(outFile, dialect='excel')

# Generates header.

outrow = []
for column in dataHeader:
  if len(outrow) == maxNumColumns: break;
  d = specs[column]

  # For simplicity sake, we just skip UUID columns in the output,
  # thus output only non uuid ones.
  if 'uuid' not in d or not d['uuid']: outrow.append(column)

outData.writerow(outrow)

# Generates data rows.
# Here we want to walk entire tree of counts, and use the keys to generate
# row values, and the leaf values as the counts saying the number of times
# resulting rows should be present in the generated data. Production algorithm
# should also sort output rows in a random order, but we do not care about the
# order for our purposes, thus skipping it.

outrow = []
f = float(numRows) / numOutRows
def generateOutput(buckets, columnId):
  d = specs[dataHeader[columnId]]

  # Skips UUID columns in the original data header.
  if 'uuid' in d and d['uuid']: return generateOutput(buckets, 1 + columnId)

  for key in buckets:
    # Deducing next value in the row out of the key.
    if d['type'] == 'enum': value = key
    else: # integer of float
      step = float(d['max'] - d['min']) / NUM_BUCKETS
      value = d['min'] + (key + 0.5) * step
      if d['type'] == 'int': value = round(value)
    outrow.append(value)

    # We reached a leaf - output generated row as much times as the count says.
    if type(buckets[key]) is int:
      for i in range(buckets[key]):
        # Randomly omits some of output rows to keep the total output size
        # about the same as input.
        if uniform(0, 1) < f:
          outData.writerow(outrow)

    # Otherwise - append the key as the next value in the generated row,
    # and go into a new recursion step.
    else: generateOutput(buckets[key], 1 + columnId)

    # Removes current value from the array.
    outrow.pop()

generateOutput(counts, 0)

print('DONE!')
