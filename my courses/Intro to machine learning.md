Из курса Kaggle "Intro to Machine Learning":

X, y - общепринято y - target, X - data on the required features
overfitting, underfitting - переобучение, недообучение

Библиотеки:
import pandas as pd - DataFrame
from sklearn.tree import DecisionTreeRegressor - Decision Tree
from sklearn.metrics import mean_absolute_error - MAE
from sklearn.model_selection import train_test_split - breaks up the data into two pieces (some to fit model, other to to validation)
form sklearn.ensemble import RandomForestRegressor - better then Decision Tree, because it's using many trees

методы DataFrame:
.describe() - all info
.head() - first n 
.dropna(axis=0) - drops missing values

методы библиотек:
DecisionTreeRegressor(random_state=n).fit(X, y) - fit model
model.predict(X) - predict for X

MAE - Mean absolute Error - средняя абсолютная ошибка (среднее по модулям ошибок "действительной" цели и предсказанной)
mean_absolute_error(y, predicted_home_prices)
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y): #function to get mae scores from different values for max_leaf_nodes
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

Example for train_test_split:
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
    melbourne_model = DecisionTreeRegressor()
    melbourne_model.fit(train_X, train_y)

    val_predictions = melbourne_model.predict(val_X)
    print(mean_absolute_error(val_y, val_predictions))