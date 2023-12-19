import numpy as np
import arviz as az
import pymc3 as pm

# Exercițiul 1
np.random.seed(42)

clusters = 3
n_cluster = [200, 150, 150]
n_total = sum(n_cluster)
means = [5, 0, -5]
std_devs = [2, 2, 2]

data = np.concatenate([np.random.normal(loc=mu, scale=sd, size=n) for mu, sd, n in zip(means, std_devs, n_cluster)])

az.plot_kde(data)
plt.show()

# Exercițiul 2
def run_gmm_model(data, num_components):
    with pm.Model() as model:
        weights = pm.Dirichlet('weights', a=np.ones(num_components))
        means = pm.Normal('means', mu=np.arange(num_components) * 5, sd=5, shape=num_components)
        sd = pm.HalfNormal('sd', sd=5)

        y_obs = pm.NormalMixture('y_obs', w=weights, mu=means, sd=sd, observed=data)

        trace = pm.sample(2000, return_inferencedata=True)

    return trace

num_components_list = [2, 3, 4]
traces = {}

for num_components in num_components_list:
    trace = run_gmm_model(data, num_components)
    traces[num_components] = trace

# Exercițiul 3
for num_components, trace in traces.items():
    waic = az.waic(trace)
    loo = az.loo(trace)
    print(f"Număr de componente: {num_components}")
    print(f"WAIC: {waic.waic}")
    print(f"LOO: {loo.loo}")
    print()

# Concluzie: Comparam valorile WAIC si LOO pentru modelele cu 2, 3 si 4 componente si alegem cel mai bun model.
