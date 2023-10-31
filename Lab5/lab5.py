import pymc3 as pm
import pandas as pd
import numpy as np

data = pd.read_csv("trafic.csv")

nr_masini = data['nr. masini'].values

with pm.Model() as model:
    lambda_ = pm.Exponential("lambda", lam=1)

    traffic = pm.Poisson("traffic", mu=lambda_, observed=nr_masini)

    trace = pm.sample(1000, tune=1000)

pm.summary(trace)

pm.plot_posterior(trace, var_names=["lambda"], credible_interval=0.95)

# Intervalul 4:00 - 7:00
lambda_0407 = trace["lambda"][(data['minut'] >= 1) & (data['minut'] <= 240)]
print("Interval 4:00 - 7:00:")
print("Medie:", lambda_0407.mean())
print("Interval de încredere 95%:", np.percentile(lambda_0407, [2.5, 97.5]))

# Intervalul 7:00 - 16:00
lambda_0716 = trace["lambda"][(data['minut'] >= 241) & (data['minut'] <= 960)]
print("Interval 7:00 - 16:00:")
print("Medie:", lambda_0716.mean())
print("Interval de încredere 95%:", np.percentile(lambda_0716, [2.5, 97.5]))

# Intervalul 16:00 - 19:00
lambda_1619 = trace["lambda"][(data['minut'] >= 961) & (data['minut'] <= 1140)]
print("Interval 16:00 - 19:00:")
print("Medie:", lambda_1619.mean())
print("Interval de încredere 95%:", np.percentile(lambda_1619, [2.5, 97.5]))

# Intervalul 19:00 - 24:00
lambda_1924 = trace["lambda"][(data['minut'] >= 1141) & (data['minut'] <= 1440)]
print("Interval 19:00 - 24:00:")
print("Medie:", lambda_1924.mean())
print("Interval de încredere 95%:", np.percentile(lambda_1924, [2.5, 97.5]))
