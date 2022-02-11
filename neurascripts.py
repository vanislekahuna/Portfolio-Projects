def binomial_logistic_regression(train_ind, train_dep, test_ind, threshold):
  """
  Will need to import the following libraries:
  import statsmodels.api as sm
  import pandas as pd 
  import numpy as np

  The model will return a new DataFrame complete with the model's sigmoid odds and predictions.
  We can then use these results with our binomial_model_accuracy() function.
  """
  x_constants = sm.add_constant(train_ind)
  log_model = sm.Logit(train_dep, x_constants)
  trained_model = log_model.fit()
  print(trained_model.summary())

  predictionData_withconstant = sm.add_constant(test_ind)
  sigmoid = trained_model.predict(predictionData_withconstant)
  test_preds = np.where(sigmoid > threshold, 1, 0)

  return sigmoid, test_preds 


def train_test_split(df, frac=0.2):
    test = df.sample(frac=frac, axis=0) # get random sample 
    train = df.drop(index=test.index) # get everything but the test sample
    return train, test


def plot_logistic(iv, dv, predictions, xaxis_lab, yaxis_lab, title):
  xpoints = np.arange(iv.min(), iv.max())

  plt.figure()
  plt.scatter(iv, dv)
  plt.scatter(iv, predictions)
  plt.grid()
  plt.xlim(iv.min(), iv.max())
  plt.xlabel(xaxis_lab)
  plt.ylabel(yaxis_lab)
  plt.title(title)
  plt.show()



def binomial_model_accuracy(prediction_train, prediction_test):
  accuracy = ( sum(prediction_train == prediction_test) / len(prediction_train) )
  return accuracy


def df_percentage(numerator, dataframe):
  percentage = round((numerator/len(dataframe)), 2) * 100
  return 
  

def chi_square(col):
  """
  For the function to work, we need to import the following libraries:
  from collections import Counter
  import numpy as np
  """
  counter = Counter(col)
  expected = len(col) * 1 / len(counter)
  chi_sq = 0
  for n in counter:
    chi_sq += ((counter[n] - expected)**2 / expected)

  return chi_sq

def plot_registration_ratios(data, column):
  """
  Plots the given column against whether they signed up for the new product.
  This function assumes data is in a dataframe and the column needs to be a string and the 'registered' column is actually in the dataframe.
  """
  total_count = data.groupby([column, 'registered'])[column].count()
  pct_contact_type = total_count / data.groupby(column)[column].count()

  pct_contact_type = pct_contact_type.unstack()
  display(pct_contact_type)

  pct_contact_type.plot(kind='barh', stacked=True)
  sns.despine()
  plt.title(f'{column} vs registered')
  plt.show()
