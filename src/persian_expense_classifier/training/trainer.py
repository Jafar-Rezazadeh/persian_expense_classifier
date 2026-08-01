# %%
from persian_expense_classifier.data.data_loader import loadData
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# %% loading data
X, Y = loadData()

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
