import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt

# Valorile observate pentru Y și θ
Y_values = [0, 5, 10]
θ_values = [0.2, 0.5]

# Creează un model PyMC
with pm.Model() as model:
    # Distribuția a priori pentru n (Poisson(10))
    n = pm.Poisson("n", mu=10)

    for Y in Y_values:
        for θ in θ_values:
            # Distribuția Binomială pentru Y
            Y_observed = pm.Binomial("Y_observed", n=n, p=θ, observed=Y)

            # Calculează distribuția a posteriori
            trace = pm.sample(1000, cores=2)

            # Vizualizează distribuția a posteriori
            pm.plot_posterior(trace, var_names=["n"])
            plt.title(f"Y = {Y}, θ = {θ}")
            plt.show()
