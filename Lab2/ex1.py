import numpy as np
from scipy import stats
from scipy.stats import expon
import matplotlib.pyplot as plt


lambda1 = 4  # Pentru primul mecanic
lambda2 = 6  # Pentru al doilea mecanic

prob_servire_primul = 0.4
prob_servire_al_doilea = 0.6

nr_clienti = 10000

timp_servire_primul = stats.expon(scale=1/lambda1).rvs(size=int(nr_clienti * prob_servire_primul))
timp_servire_al_doilea = stats.expon(scale=1/lambda2).rvs(size=int(nr_clienti * prob_servire_al_doilea))

timp_servire_total = np.concatenate([timp_servire_primul, timp_servire_al_doilea])

media_servire = np.mean(timp_servire_total)
deviatia_standard_servire = np.std(timp_servire_total)

print(f"Media timpului de servire: {media_servire}")
print(f"Deviația standard a timpului de servire: {deviatia_standard_servire}")

plt.hist(timp_servire_total, bins=50, density=True, alpha=0.6, color='b', label='Distribuție de servire')
plt.xlabel('Timpul de servire')
plt.ylabel('Densitatea de probabilitate')
plt.legend(loc='upper right')
plt.title('Densitatea distribuției timpului de servire')
plt.show()