# Analiza Bayesiană cu PyMC3

Acest repositoriu conține un exemplu de analiză Bayesiană folosind PyMC3 pentru a calcula distribuția a posteriori pentru n, dată o distribuție prior pentru n și observații pentru Y în funcție de θ. Vom discuta efectele lui Y și θ asupra distribuției a posteriori în acest fișier README.md.

## Efectul lui Y

Efectul lui Y asupra distribuției a posteriori pentru n este notabil prin faptul că cu cât valoarea observată a lui Y este mai mare, cu atât distribuția a posteriori a lui n se va deplasa spre valori mai mari. Acest lucru se datorează faptului că, cu mai mulți clienți care cumpără produsul, estimarea parametrului n, care reprezintă numărul total de clienți, va fi mai mare pentru a se potrivi datelor observate.

## Efectul lui θ

Efectul lui θ asupra distribuției a posteriori pentru n este legat de probabilitatea ca un client să cumpere produsul. Cu cât valoarea lui θ este mai mare (adică probabilitatea de cumpărare este mai mare), cu atât distribuția a posteriori pentru n se va deplasa spre valori mai mari. Acest lucru se datorează faptului că o probabilitate mai mare de cumpărare implică un număr mai mare de clienți pentru a obține rezultatele observate.

