import random

filename = 'uf20-01.cnf'

with open(filename,'r') as file:
     all_lines = file.readlines()


def initial_solution(n_vars):
    solution = np.zeros(n_vars)
    for variable in solution:
        variable = randint(0, 1)
    return solution


def to_tuple(n):
    n = n - 1
    if n < 0:
        return (-1*n, 0)
    else:
        return (n, 1)


def evaluate(clause, solution):
    sucesses = []
    for position, value in clause:
        sucesses.append(solution[position] == value)
    return sucesses.any()


for line in all_lines:
    if line.startswith('p'):
        print(line)
        a, b, n_vars, n_cla = line.split(' ')
        solution = initial_solution(n_vars)

    clauses = []
    if not line.startswith('c'):
        v1, v2, v3, _ = line.split(' ')
        clauses.append(to_tuple(v1), to_tuple(v2), to_tuple(v3))

################ IDEAS ################
# Avaliar solucao
count = 0
for clause in clauses:
    if evaluate(clause, solution):
        count += 1

# Perturbar solucao
disturbance = randint(0, n_vars-1)
solution[disturbance] = 1 - solution[disturbance]

# clause = [(1,1), (2,0)]

# solution = [1, 1, 1, 1, ...]
