import numpy as np
import matplotlib.pyplot as plt

# Probabilitățile corecte
prob_ss = 0.7 * 0.7  # Probabilitatea de a obține două stele (stema)
prob_sb = 2 * 0.3 * 0.7  # Probabilitatea de a obține o stea și o banană
prob_bs = 2 * 0.7 * 0.3  # Probabilitatea de a obține o banană și o stea
prob_bb = 0.3 * 0.3  # Probabilitatea de a obține două banane

nr_experimente = 100
rezultate_experimente = []

for _ in range(nr_experimente):
    rezultat_experiment = np.random.choice(['ss', 'sb', 'bs', 'bb'], size=10, p=[prob_ss, prob_sb, prob_bs, prob_bb])
    rezultate_experimente.append(rezultat_experiment)

distributie_ss = [np.sum([1 for rezultat in rezultate_experiment if rezultat == 'ss']) for rezultate_experiment in rezultate_experimente]
distributie_sb = [np.sum([1 for rezultat in rezultate_experiment if rezultat == 'sb']) for rezultate_experiment in rezultate_experimente]
distributie_bs = [np.sum([1 for rezultat in rezultate_experiment if rezultat == 'bs']) for rezultate_experiment in rezultate_experimente]
distributie_bb = [np.sum([1 for rezultat in rezultate_experiment if rezultat == 'bb']) for rezultate_experiment in rezultate_experimente]

plt.hist(distributie_ss, bins=range(11), alpha=0.6, label='ss')
plt.hist(distributie_sb, bins=range(11), alpha=0.6, label='sb')
plt.hist(distributie_bs, bins=range(11), alpha=0.6, label='bs')
plt.hist(distributie_bb, bins=range(11), alpha=0.6, label='bb')
plt.xlabel('Numărul de apariții în 10 aruncări')
plt.ylabel('Frecvență')
plt.legend(loc='upper right')
plt.title('Distribuțiile rezultatelor în cele 10 aruncări ale celor două monede')
plt.show()
