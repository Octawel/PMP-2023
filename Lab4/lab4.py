import numpy as np
from scipy.stats import norm
import scipy.optimize as opt
import matplotlib.pyplot as plt

# Exercițiul 1: Vizualizarea distribuțiilor

lambda_clienti = 20  # Parametrul distribuției Poisson (clienți pe oră)
media_plata = 2  # Media distribuției normale pentru timpul de plată (minute)
deviatie_standard_plata = 0.5  # Deviația standard a distribuției normale pentru timpul de plată (minute)
media_pregatire = 3  # Media distribuției exponențiale pentru timpul de pregătire (minute)

numar_clienti = np.random.poisson(lambda_clienti, 1000)
timp_plata = np.random.normal(media_plata, deviatie_standard_plata, 1000)
timp_pregatire = np.random.exponential(media_pregatire, 1000)

# Exercițiul 2: Calcularea timpului mediu de așteptare

def timp_servire(alfa, numar_clienti, media_plata, deviatie_standard_plata):
    timp_pregatire = np.random.exponential(alfa, numar_clienti)
    timp_total = timp_pregatire + np.random.normal(media_plata, deviatie_standard_plata, numar_clienti)
    return timp_total

numar_clienti_pe_ora = 20
media_plata = 2
deviatie_standard_plata = 0.5

alfa_maxim = 4.4 

numar_simulari = 10000  # Numărul de simulări Monte Carlo

timp_total_servire = np.zeros(numar_simulari)

for i in range(numar_simulari):
    timp_total_servire[i] = np.mean(timp_servire(alfa_maxim, numar_clienti_pe_ora, media_plata, deviatie_standard_plata))

timp_mediu_asteptare = np.mean(timp_total_servire)
print(f"Timpul mediu de așteptare pentru a fi servit al unui client este de aproximativ {timp_mediu_asteptare:.2f} minute.")

# Exercițiul 3: Găsirea valorii maxime a lui α

def timp_total_servire(alpha, numar_clienti_pe_ora, media_plata, deviatie_standard_plata):
    timp_pregatire = np.random.exponential(alpha, numar_clienti_pe_ora)
    timp_plata = np.random.normal(media_plata, deviatie_standard_plata, numar_clienti_pe_ora)
    return np.sum(timp_pregatire) + np.sum(timp_plata)

def probabilitate_servire_sub_limita(alpha, numar_clienti_pe_ora, media_plata, deviatie_standard_plata, timp_total_maxim):
    numar_simulari = 10000
    servire_sub_limita = 0
    for _ in range(numar_simulari):
        timp_total = timp_total_servire(alpha, numar_clienti_pe_ora, media_plata, deviatie_standard_plata)
        if timp_total <= timp_total_maxim:
            servire_sub_limita += 1
    probabilitate = servire_sub_limita / numar_simulari
    return probabilitate

timp_total_maxim = 15
numar_clienti_pe_ora = 20
media_plata = 2
deviatie_standard_plata = 0.5

rezultat = opt.root_scalar(probabilitate_servire_sub_limita, bracket=[0.1, 10.0], args=(numar_clienti_pe_ora, media_plata, deviatie_standard_plata, timp_total_maxim), method='bisect')
alfa_maxim = rezultat.root

print(f"Cea mai mare valoare a lui α pentru a servi clienții în mai puțin de 15 minute cu o probabilitate de 95% este {alfa_maxim:.2f} minute.")
