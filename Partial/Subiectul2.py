import numpy as np
# import matplotlib.pyplot as plt
# import pymc3 as pm

mu = 10  # Media
sigma = 2  # Deviația standard

timpi_asteptare = np.random.normal(mu, sigma, 200)

print(timpi_asteptare)


# PyMC

# plt.hist(timpi_asteptare, bins=20, density=True, alpha=0.6, color='b')
# plt.title('Distribuția simulată a timpilor medii de așteptare')
# plt.xlabel('Timp mediu de așteptare')
# plt.ylabel('Densitatea de probabilitate')
# plt.show()

# model = pm.Model()
# with model:
#     # Alegerea unei distribuții normale ca a priori pentru parametrul mediu (p)
#     p = pm.Normal('p', mu=5, sd=2)

# pm.model_to_graphviz(model)

# with model:
#     observed_data = pm.Normal('observed_data', mu=p, sd=2, observed=timpi_asteptare)

# with model:
#     trace = pm.sample(1000, tune=1000, cores=1) 

# pm.plot_posterior(trace)
# plt.title('Distribuția a posteriori pentru parametrul mediu')
# plt.show()

