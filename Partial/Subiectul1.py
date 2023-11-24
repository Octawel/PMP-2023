import numpy as np

from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
import pandas as pd


def arunca_moneda(masluita=False, prob_stema=1/3):
    if masluita and np.random.rand() < prob_stema:
        return 'stema'
    return 'pajura' if np.random.rand() < 0.5 else 'stema'

numar_jocuri = 20000
cistiguri = {'P0': 0, 'P1': 0}

for _ in range(numar_jocuri):
    incepe_cu_P0 = np.random.rand() < 0.5
    
    steme_P0 = 1 if arunca_moneda(True) == 'stema' else 0
    steme_P1 = 1 if arunca_moneda() == 'stema' else 0
    
    if incepe_cu_P0:
        for _ in range(steme_P1 + 1):
            steme_P0 += 1 if arunca_moneda(True) == 'stema' else 0
    else:
        for _ in range(steme_P0):
            steme_P1 += 1 if arunca_moneda() == 'stema' else 0
    
    cistigator = 'P0' if steme_P0 > steme_P1 else 'P1'
    cistiguri[cistigator] += 1

print(f'Numărul de jocuri câștigate de P0: {cistiguri["P0"]}')
print(f'Numărul de jocuri câștigate de P1: {cistiguri["P1"]}')
print(f'Procentajul de jocuri câștigate de P0: {cistiguri["P0"]/numar_jocuri*100:.2f}%')
print(f'Procentajul de jocuri câștigate de P1: {cistiguri["P1"]/numar_jocuri*100:.2f}%')


model = BayesianModel([('P0', 'Stema_P0'), ('P0', 'Stema_P1'), ('P1', 'Stema_P1')])

data = []
for _ in range(numar_jocuri):
    incepe_cu_P0 = np.random.rand() < 0.5
    steme_P0 = 1 if arunca_moneda(True) == 'stema' else 0
    steme_P1 = 1 if arunca_moneda() == 'stema' else 0
    if incepe_cu_P0:
        for _ in range(steme_P1 + 1):
            steme_P0 += 1 if arunca_moneda(True) == 'stema' else 0
    else:
        for _ in range(steme_P0):
            steme_P1 += 1 if arunca_moneda() == 'stema' else 0
    data.append({'P0': steme_P0, 'P1': steme_P1, 'Stema_P0': int(steme_P0 > steme_P1), 'Stema_P1': steme_P1})

df_data = pd.DataFrame(data)

model.fit(df_data, estimator=MaximumLikelihoodEstimator)

from pgmpy.inference import VariableElimination
inference = VariableElimination(model)
result = inference.query(variables=['Stema_P0'], evidence={'Stema_P1': 0})
print(result)

# Import clasele și funcțiile necesare din biblioteca pgmpy pentru construirea și antrenarea modelului Bayesian.
# Definesc o funcție arunca_moneda care simulează aruncarea unei monede. Argumentele opționale masluita și prob_stema controlează dacă moneda este măsluită și probabilitatea de a obține stema.
# Simulez desfășurarea a 20,000 de jocuri între jucătorii P0 și P1.
# Tin evidența câștigurilor pentru fiecare jucător într-un dicționar numit cistiguri.
# Afișez rezultatele astfel: numărul de jocuri câștigate de P0, numărul de jocuri câștigate de P1, procentajul de jocuri câștigate de P0 și procentajul de jocuri câștigate de P1.
# Definesc o rețea Bayesiană cu nodurile și conexiunile specifice jocului.
# Colectez datele simulate într-un obiect DataFrame.
# Antrenez modelul folosind datele simulate prin metoda Maximum Likelihood.
# Realizez inferența pentru a estima probabilitatea ca fața monedei în prima rundă să fie stema, având informații despre faptul că în a doua rundă nu s-a obținut nicio stema.
# Afișez rezultatul estimării (Stema_P0(0) = Nu obtinem stema pentru prima runda, Stema_P0(1) = Obtinem stema pentru prima runda)