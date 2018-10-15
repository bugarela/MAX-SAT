from random import randint
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns

filename = 'uf20-01.cnf'
max_iterations = 250000
t0 = 100
tf = 1

with open(filename,'r') as file:
     all_lines = file.readlines()


def initial_solution(n_vars):
    solution = []
    for _ in range(n_vars):
        solution.append(randint(0, 1))
    return solution


def to_tuple(n_str):
    n = int(n_str)
    if n < 0:
        return (-1*(n+1), 0)
    else:
        return (n-1, 1)


def evaluate(clause, solution):
    sucesses = []
    for position, value in clause:
        sucesses.append(solution[position] == value)
    return any(sucesses)


def evaluate_all(clauses, solution):
    count = 0
    for clause in clauses:
        if evaluate(clause, solution):
            count += 1
    return count


def random_search(clauses, solution):
    new_solution = solution
    score = evaluate_all(clauses, solution)
    iterations = 0

    scores = []

    while iterations < max_iterations:
        iterations += 1
        disturbance = randint(0, len(solution)-1)
        new_solution[disturbance] = 1 - solution[disturbance]
        new_score = evaluate_all(clauses, new_solution)
        if new_score >= score:
            solution = new_solution
            score = new_score
        else:
            new_solution = solution
        scores.append(score)
    print(len(np.arange(max_iterations)), len(scores))
    sns.lineplot(x=np.arange(max_iterations), y=scores)
    plt.savefig('random_search_convergence.png')

    return solution, score


def next_temperature(i):
    return t0*(tf/t0)**(i/max_iterations)


def simmulated_annealing(clauses, solution):
    ## TO DO
    new_solution = solution
    temperature = t0
    score = evaluate_all(clauses, solution)
    iterations = 0

    scores = []

    while iterations < max_iterations:
        iterations += 1
        disturbance = randint(0, len(solution)-1)
        new_solution[disturbance] = 1 - solution[disturbance]
        new_score = evaluate_all(clauses, new_solution)
        delta = new_score - score
        if delta <= 0 or rand() < exp(-delta/temperature):
            solution = new_solution
            score = new_score
        scores.append(score)
        temperature = next_temperature(iterations)

    print(len(np.arange(max_iterations)), len(scores))
    sns.lineplot(x=np.arange(max_iterations), y=scores)
    plt.savefig('simmulated_annealing_convergence.png')

    return solution, score

clauses = []
for line in all_lines:
    if line.startswith('p'):
        _, _, n_vars, n_cla = line.split()
        n_vars = int(n_vars)
        n_cla = int(n_cla)

    elif line.startswith('%'):
        break

    elif not line.startswith('c'):
        v1, v2, v3, _ = line.split()
        clauses.append([to_tuple(v1), to_tuple(v2), to_tuple(v3)])

solution, score = random_search(clauses, initial_solution(n_vars))
print(solution, score)
