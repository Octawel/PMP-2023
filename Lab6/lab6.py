import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt

Y_values = [0, 5, 10]
θ_values = [0.2, 0.5]

with pm.Model() as model:
    n = pm.Poisson("n", mu=10)

    for Y in Y_values:
        for θ in θ_values:
            Y_observed = pm.Binomial("Y_observed", n=n, p=θ, observed=Y)

            # Calculează distribuția a posteriori
            trace = pm.sample(1000, cores=2)

            pm.plot_posterior(trace, var_names=["n"])
            plt.title(f"Y = {Y}, θ = {θ}")
            plt.show()
