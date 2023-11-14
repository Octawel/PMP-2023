import pandas as pd
import pymc3 as pm
import matplotlib.pyplot as plt

# Incarcarea setului de date
df = pd.read_csv('auto-mpg.csv')

# Extragerea doar a coloanelor de interes
df = df[['horsepower', 'mpg']]

# Conversia coloanei 'horsepower' la tip numeric, tratând '?' ca missing values
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')

# Eliminarea rândurilor cu missing values
df = df.dropna()

# Vizualizare relația de dependență dintre horsepower și mpg
plt.scatter(df['horsepower'], df['mpg'])
plt.xlabel('Horsepower')
plt.ylabel('MPG')
plt.title('Relația dintre Horsepower și MPG')
plt.show()

# Definirea modelului în PyMC3
with pm.Model() as model:
    # Prior pentru intercept și coeficientul de horsepower
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=10)

    # Modelul de regresie
    mu = alpha + beta * df['horsepower']

    # Likelihood
    sigma = pm.HalfNormal('sigma', sd=10)
    mpg = pm.Normal('mpg', mu=mu, sd=sigma, observed=df['mpg'])

# Determinarea dreptei de regresie
with model:
    trace = pm.sample(1000, tune=1000)

# Afișarea rezultatelor
pm.summary(trace).round(2)

# Afișarea dreptei de regresie și intervalului de confidență
plt.scatter(df['horsepower'], df['mpg'])
plt.xlabel('Horsepower')
plt.ylabel('MPG')

pm.plot_posterior_predictive_glm(trace, samples=100, eval=np.linspace(df['horsepower'].min(), df['horsepower'].max(), 100), color='blue', alpha=0.3)
plt.title('Dreapta de regresie și intervalul de confidență 95%')
plt.show()
