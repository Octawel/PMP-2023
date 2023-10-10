import random
import matplotlib.pyplot as plt

def generate_experiment():
    result_counts = {'ss': 0, 'sb': 0, 'bs': 0, 'bb': 0}

    for i in range(10):
        coin1 = 's' if random.random() < 0.5 else 'b'
        coin2 = 's' if random.random() < 0.3 else 'b'

        result_counts[coin1 + coin2] += 1

    return result_counts

experiment_results = {'ss': 0, 'sb': 0, 'bs': 0, 'bb': 0}
num_experiments = 100

for i in range(num_experiments):
    experiment = generate_experiment()
    for key in experiment_results:
        experiment_results[key] += experiment[key]

labels = experiment_results.keys()
counts = [experiment_results[key] for key in labels]

plt.bar(labels, counts)
plt.xlabel('Rezultat')
plt.ylabel('Număr de apariții')
plt.title('Distribuția rezultatelor în 100 de experimente cu 10 aruncări')
plt.show()