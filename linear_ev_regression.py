# %% read data
import pandas as pd
from seaborn import regression
from sklearn import preprocessing
import numpy as np
import seaborn as sns

train = pd.read_csv("traincap.csv")
test = pd.read_csv("testcap.csv")

# %% to make a single dataset (adding rows together) 
# For this regression models purpose, actually there was no need to make seperate train and test model, these were created for trial with other model. So, by following code, the two sets are again being brought to one.
df = [train, test]
df = pd.concat(df)
# %% reindex as follows. Because they the dataset is now divded 0-116 and then again 0-116 instead of 0-234

df.index = np.arange(df.shape[0])

# %%Following the video AK Python
from sklearn import linear_model

# %% Creating dataframes
inputs = df.drop(
    [
        "Transmission",
        "Battery over 15 kWh ?",
        "BEV, PHEV > 15 kWh or PHEV < 15 kWh",
        "Cylinders",
        "Engine size (L)",
        "Fuel Type 1",
        "Fuel Type 2",
        "Incentive available",
        "Purchase",
        "12 m Lease",
        "24 m Lease",
        "36 m Lease",
        "CO2  Rating",
        "Cost per 100km ($)",
        "Max Incentive ($)",
        "Motor(KW)",
        "Range electricity (km)",
        "Range fuel (km)",
        "Smog Rating",
        "Price($)",
        "Year Make Model Fuel",
    ],
    axis="columns",
)

# %% creating targets
target = df["Price($)"]


# %% Converting string values to numerical values
from sklearn.preprocessing import LabelEncoder

Numerics = LabelEncoder()


# %%# %% Converting string values to numerical values
inputs["BEV or PHEV_n"] = Numerics.fit_transform(inputs["BEV or PHEV"])
inputs["Vehicle Class_n"] = Numerics.fit_transform(inputs["Vehicle Class"])
inputs["Make_n"] = Numerics.fit_transform(inputs["Make"])
inputs["Model_n"] = Numerics.fit_transform(inputs["Model"])


# %%Drop string columns
inputs_n = inputs.drop(
    ["BEV or PHEV", "Vehicle Class", "Make", "Model"],
    axis="columns",
)
inputs_n

# %% linear regression model
model = linear_model.LinearRegression()


# %%Training
model.fit(inputs_n, target)

# %% Prediction one value (set the values sequentially)
pred = model.predict([[2020, 3.00, 5, 567, 100, 50, 2, 6, 2, 13]])
print(pred)


# %% Prediction all prices
pred = model.predict(inputs_n)
print(pred)


# %% Showing actual price and predicted price side by side
pdf = pd.DataFrame({"Actual": df["Price($)"], "Predicted": pred})
pdf


# %% Check regression scores for the model and coefficient
x = inputs_n
y = pred
# (see above cells to find what's pred)
r_sq = model.score(x, y)
print("coefficient of determination:", r_sq)
print("intercept:", model.intercept_)
print("slope:", model.coef_)


# %% extra: converting array into dataframe/table; here pred is now an array, can be converted to table by following
import numpy as np
import pandas as pd

preddf = pd.DataFrame(pred)

# %% merging the inputs and the new predicted value together
data = inputs_n.join(target)
data = data.join(preddf)
# %%rename the new column for preddf currently named as 0.
data.rename(columns={0: "Predicted Price($)"}, inplace=True)


# %% Plotting the results on a regression line (Not the ideal way)

import matplotlib.pyplot as plt

x = data["Total range(km)"]

y = data["Predicted Price($)"]
plt.plot(x, y, "o")


m, b = np.polyfit(x, y, 1)

sns.set(font_scale=2)
plt.xlabel("Total range(km)", size=20)
plt.ylabel("Predicted Price($)", size=20)
plt.title("Correlation of range of cars (km) with price($)", size=30)
plt.plot(x, m * x + b)
plt.figure(figsize=(20, 14))

# %%# %% Another way of plotting the results on a regression line (Not the ideal way)
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(font_scale=4)
g = sns.regplot(
    x=data["Total range(km)"],
    y=data["Predicted Price($)"],
    line_kws={"color": "r", "alpha": 0.9, "lw": 5},
)
g.figure.set_size_inches(18.5, 10.5)
plt.xlabel("Total range(km)", size=30)
plt.ylabel("Predicted Price($)", size=30)
plt.title("Correlation of range of cars (km) with price($)", size=40)
sns.set(font_scale=4)
sns.despine()
plt.show()


# %%# %% Another preditor)
import seaborn as sns
import matplotlib.pyplot as plt

g = sns.regplot(
    x=data["Recharge Time(h)"],
    y=data["Predicted Price($)"],
    line_kws={"color": "r", "alpha": 0.9, "lw": 5},
)
g.figure.set_size_inches(18.5, 10.5)
plt.xlabel("Recharge Time(h)", size=30)
plt.ylabel("Predicted Price($)", size=30)
plt.title("Correlation of recharge time(h) with price($)", size=40)
sns.set(font_scale=4)
sns.despine()
plt.show()


# %%# %% Another preditor)
import seaborn as sns
import matplotlib.pyplot as plt

g = sns.regplot(
    x=data["CO2 Emissions(g/km)"],
    y=data["Predicted Price($)"],
    line_kws={"color": "r", "alpha": 0.9, "lw": 5},
)
g.figure.set_size_inches(18.5, 10.5)
plt.xlabel("CO2 Emissions(g/km)", size=30)
plt.ylabel("Predicted Price($)", size=30)
plt.title("Correlation of CO2 Emissions(g/km) with price($)", size=40)
sns.set(font_scale=4)
sns.despine()
plt.show()
# %%

# %%# %% Another preditor)
import seaborn as sns
import matplotlib.pyplot as plt

g = sns.regplot(
    x=data["Total capacity Battery life(KWH)"],
    y=data["Predicted Price($)"],
    line_kws={"color": "r", "alpha": 0.9, "lw": 5},
)
g.figure.set_size_inches(18.5, 10.5)
plt.xlabel("Total capacity Battery life(KWH)", size=30)
plt.ylabel("Predicted Price($)", size=30)
plt.title("Correlation of Total capacity Battery life(KWH) with price($)", size=40)
sns.set(font_scale=4)
sns.despine()
plt.show()


# %% Violin plot for vehicle classes [extra thing, not related to regression]
import matplotlib.pyplot as plt
import seaborn as sns

# data
data = data

# plot figure
plt.figure(figsize=(20, 16))
p = sns.violinplot(x=df["Vehicle Class"], y="Price($)", data=data)

# get label text
_, ylabels = plt.yticks()
_, xlabels = plt.xticks()
plt.xticks(rotation=45)
sns.set(font_scale=2)

plt.show()


# %% Using an advance linear regression model (Statsmodels)
import numpy as np
import statsmodels.api as sm


# %% Don't forget that here we used the previous data preparation, converting to numeric etc.
x = inputs_n
y = target
x, y = np.array(x), np.array(y)

# %%That’s how you add the column of ones to x with add_constant(). It takes the input array x as an argument and returns a new array with the column of ones inserted at the beginning.
x = sm.add_constant(x)

# %% Create a model and fit it. This model is called OLS model. Ordinary Least Squares Regression

model = sm.OLS(y, x)

# %%Once your model is created, you can apply .fit() on it:
# By calling .fit(), you obtain the variable results, which is an instance of the class statsmodels.regression.linear_model.RegressionResultsWrapper. This object holds a lot of information about the regression model
results = model.fit()

# %%  Get results, see the scores/coefficient standard errors, etc.

print(results.summary())


# %%coefficent of the fit check
results.params


# # R-squared and the Goodness-of-Fit
# R-squared evaluates the scatter of the data points around the fitted regression line. It is also called the coefficient of determination, or the coefficient of multiple determination for multiple regression. For the same data set, higher R-squared values represent smaller differences between the observed data and the fitted values.

# R-squared is the percentage of the dependent variable variation that a linear model explains.

# {\displaystyle R^2 = \frac {\text{Variance explained by the model}}{\text{Total variance}}}
# R-squared is always between 0 and 100%:

# 0% represents a model that does not explain any of the variation in the response variable around its mean. The mean of the dependent variable predicts the dependent variable as well as the regression model.
# 100% represents a model that explains all the variation in the response variable around its mean.


# ******* details can be found here: https://www.datarobot.com/blog/ordinary-least-squares-in-python/


# %% not sure if it needed or not
%pylab inline


# %% Visualization of the various predictors' relation

# Plot predicted values
fix, ax = plt.subplots()
ax.scatter(inputs_n["Recharge Time(h)"], results.predict(), alpha=0.5,
        label='predicted')
# Plot observed values
ax.scatter(inputs_n["Recharge Time(h)"], df['Price($)'], alpha=0.5,
        label='observed')

import seaborn as sns
x= inputs_n["Recharge Time(h)"]
y =results.predict()
m, b = np.polyfit(x, y, 1)

sns.set(font_scale=2)
plt.xlabel("Recharge Time(h)", size=20)
plt.ylabel("Predicted Price($)", size=20)
plt.title("Correlation of Recharge Time(h) with price($)", size=30)
plt.plot(x, m * x + b)
plt.figure(figsize=(20, 14))





# %%# Plot predicted values
fix, ax = plt.subplots()
ax.scatter(inputs_n["Total capacity Battery life(KWH)"], results.predict(), alpha=0.5,
        label='predicted')
# Plot observed values
ax.scatter(inputs_n["Total capacity Battery life(KWH)"], df['Price($)'], alpha=0.5,
        label='observed')

import seaborn as sns
x= inputs_n["Total capacity Battery life(KWH)"]
y =results.predict()
m, b = np.polyfit(x, y, 1)

sns.set(font_scale=2)
plt.xlabel("Total capacity Battery life(KWH)", size=20)
plt.ylabel("Predicted Price($)", size=20)
plt.title("Correlation of Total capacity Battery life(KWH) with price($)", size=30)
plt.plot(x, m * x + b)
plt.figure(figsize=(20, 14))



# %%# Plot predicted values
fix, ax = plt.subplots()
ax.scatter(inputs_n["Total range(km)"], results.predict(), alpha=0.5,
        label='predicted')
# Plot observed values
ax.scatter(inputs_n["Total range(km)"], df['Price($)'], alpha=0.5,
        label='observed')

import seaborn as sns
x= inputs_n["Total range(km)"]
y =results.predict()
m, b = np.polyfit(x, y, 1)

sns.set(font_scale=2)
plt.xlabel("Total range(h)", size=20)
plt.ylabel("Predicted Price($)", size=20)
plt.title("Correlation of Total range(km) with price($)", size=30)
plt.plot(x, m * x + b)
plt.figure(figsize=(20, 14))



# %%# Plot predicted values
fix, ax = plt.subplots()
ax.scatter(inputs_n["Seating Capacity"], results.predict(), alpha=0.5,
        label='predicted')
# Plot observed values
ax.scatter(inputs_n["Seating Capacity"], df['Price($)'], alpha=0.5,
        label='observed')

import seaborn as sns
x= inputs_n["Seating Capacity"]
y =results.predict()
m, b = np.polyfit(x, y, 1)

sns.set(font_scale=2)
plt.xlabel("Seating Capacity", size=20)
plt.ylabel("Predicted Price($)", size=20)
plt.title("Correlation of Seating Capacity with price($)", size=30)
plt.plot(x, m * x + b)
plt.figure(figsize=(20, 14))



# %%
