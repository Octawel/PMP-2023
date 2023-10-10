import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

alpha = [4, 4, 5, 5]  # Parametrii α pentru cele patru distribuții Gamma
lambda_ = [3, 2, 2, 3]  # Parametrii λ pentru cele patru distribuții Gamma
latenta = stats.expon(scale=1/4)  # Distribuția exponențială pentru latență

prob_directie = [0.25, 0.25, 0.30, 0.20]

nr_clienti = 10000
timp_servire = []

for _ in range(nr_clienti):
    server_ales = np.random.choice([0, 1, 2, 3], p=prob_directie)
    timp_procesare_server = stats.gamma(alpha[server_ales], scale=1/lambda_[server_ales]).rvs()
    timp_latenta = latenta.rvs()
    timp_total = timp_procesare_server + timp_latenta
    timp_servire.append(timp_total)

prob_timp_mai_mare_de_3 = np.sum(np.array(timp_servire) > 3) / nr_clienti

print(f"Probabilitatea ca timpul necesar pentru servire să fie mai mare de 3 ms: {prob_timp_mai_mare_de_3}")

plt.hist(timp_servire, bins=50, density=True, alpha=0.6, color='b', label='Distribuție de servire')
plt.xlabel('Timpul necesar pentru servire (ms)')
plt.ylabel('Densitatea de probabilitate')
plt.legend(loc='upper right')
plt.title('Densitatea distribuției timpului necesar pentru servire')
plt.show()
