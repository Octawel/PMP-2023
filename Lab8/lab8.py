import pymc3 as pm
import pandas as pd
import numpy as np

data = pd.read_csv('Prices.csv')

data['log_disk'] = np.log(data['hard_disk'])

with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta1 = pm.Normal('beta1', mu=0, sd=10)
    beta2 = pm.Normal('beta2', mu=0, sd=10)
    sigma = pm.HalfNormal('sigma', sd=1)

    mu = alpha + beta1 * data['processor_freq'] + beta2 * data['log_disk']

    price = pm.Normal('price', mu=mu, sd=sigma, observed=data['price'])

    trace = pm.sample(1000, tune=1000)

hdi_beta1 = pm.stats.hpd(trace['beta1'])
hdi_beta2 = pm.stats.hpd(trace['beta2'])

# computer cu frecvența de 33 MHz și hard disk de 540 MB
processor_freq_new = 33
log_disk_new = np.log(540)

with model:
    post_pred = pm.sample_posterior_predictive(trace, samples=5000)

price_pred = post_pred['price']

# intervalul de 90% HDI 
hdi_price = pm.stats.hpd(price_pred)

with model:
    post_pred_full = pm.sample_posterior_predictive(trace, samples=5000, var_names=['price'])

hdi_prediction = pm.stats.hpd(post_pred_full['price'])

print(f'Intervalul de 95% HDI pentru beta1: {hdi_beta1}')
print(f'Intervalul de 95% HDI pentru beta2: {hdi_beta2}')
print(f'Intervalul de 90% HDI pentru prețul așteptat: {hdi_price}')
print(f'Intervalul de 90% HDI pentru prețul de predicție: {hdi_prediction}')
