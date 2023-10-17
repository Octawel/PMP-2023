from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# 1. Definirea modelului structurii. Putem defini rețeaua specificând doar o listă de muchii.
model = BayesianNetwork([('Cutremur', 'Incendiu'), ('Cutremur', 'AlarmăIncendiu'), ('Incendiu', 'AlarmăIncendiu')])

# Definirea CPD-urilor individuale.
cpd_cutremur = TabularCPD(variable='Cutremur', variable_card=2, values=[[0.9995], [0.0005]])
cpd_incendiu = TabularCPD(variable='Incendiu', variable_card=2, 
                          values=[[0.01, 0.03],  # Probabilitatea incendiului fără și cu cutremur
                                  [0.99, 0.97]], # Probabilitatea incendiului fără și cu cutremur
                          evidence=['Cutremur'],
                          evidence_card=[2])
cpd_alarmă_incendiu = TabularCPD(variable='AlarmăIncendiu', variable_card=2, 
                                 values=[[0.9999, 0.02, 0.95, 0.98],
                                         [0.0001, 0.98, 0.05, 0.02]],
                                 evidence=['Cutremur', 'Incendiu'],
                                 evidence_card=[2, 2])

# Asocierea CPD-urilor cu rețeaua
model.add_cpds(cpd_cutremur, cpd_incendiu, cpd_alarmă_incendiu)

# Verificarea modelului
assert model.check_model()

# Realizarea inferenței exacte folosind Variable Elimination
infer = VariableElimination(model)

# 2. Probabilitatea ca un cutremur să fi avut loc, dacă alarma de incendiu a fost declanșată.
result_cutremur = infer.query(variables=['Cutremur'], evidence={'AlarmăIncendiu': 1})
print("1. Probabilitatea că a avut loc un cutremur dacă alarma de incendiu a fost declanșată:")
print(result_cutremur)

# 3. Probabilitatea ca un incendiu să fi avut loc fără ca alarma de incendiu să se activeze.
result_incendiu = infer.query(variables=['Incendiu'], evidence={'AlarmăIncendiu': 0})
print("\n2. Probabilitatea ca un incendiu să fi avut loc fără ca alarma de incendiu să se activeze:")
print(result_incendiu)
