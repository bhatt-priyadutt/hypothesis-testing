from statsmodels.api import OLS
import statsmodels.api as sm
import math
import pandas as pd

df = pd.read_csv('test.csv')

x_with_constant = sm.add_constant(df[['x']])
#with constant
ols_with_constant = OLS(df['y'], x_with_constant)
ols_fit_with_constant = ols_with_constant.fit()
pred_df_with_constant = pd.DataFrame()
pred_df_with_constant["y_pred"] = ols_fit_with_constant.predict(x_with_constant)
pred_df_with_constant["y_actual"] = df['y']

print("Model Summary: ",ols_fit_with_constant.summary())

se = math.sqrt((1 / (len(df)-2)) * (sum(((df['y'] - pred_df_with_constant['y_pred'])**2)) / sum(((df['x'] - df['x'].mean())**2))))

b1 = 0.4446
print("T-test statistic: ",b1 / se)

