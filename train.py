import pandas as pd

df = pd.read_csv("data/housing.csv")

print(df.head())

print(df.shape)
print(df.columns)
print(df.info())
print(df.isnull().sum())
df = df.dropna(subset=['Price'])
df['Garage'] = df['Garage'].fillna(df['Garage'].mode()[0])
print(df.isnull().sum())
X = df.drop('Price', axis=1)
y = df['Price']
X = pd.get_dummies(X, drop_first=True)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
from sklearn.metrics import mean_absolute_error, r2_score

print("MAE:", mean_absolute_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))
print(df.corr(numeric_only=True)['Price'].sort_values(ascending=False))
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("R2:", r2_score(y_test, pred))
print(df['Price'].describe())

df['Price'] = df['Area'] * 3000 + df['Bedrooms'] * 50000
