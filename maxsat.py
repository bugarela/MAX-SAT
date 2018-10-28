from random import randint, random, choice
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set_style("darkgrid")

instances = ['uf20-91/uf20-01',
             'uf100-430/uf100-01',
             'uf250-1065/uf250-01']

palettes = ['PuOr', 'PRGn', 'Spectral']

INSTANCE = 2
max_iterations = 250000
t0 = 5
tf = 0.001

instance = instances[INSTANCE]
sns.set_palette(palettes[INSTANCE])
results = pd.DataFrame()


def next_temperature(i):
    return (t0-tf)/(np.cosh(10*i/max_iterations)) + tf


def initial_solution(n_vars):
    return [choice([True, False]) for _ in range(n_vars)]


def to_tuple(n_str):
    n = int(n_str)
    return (abs(n)-1, n >= 0)


def evaluate(clause, solution):
    return any([solution[position] == value for position, value in clause])


def evaluate_all(clauses, solution):
    return np.sum([evaluate(clause, solution) for clause in clauses])


def disturb(solution):
    return [1 - var if random() < 0.01 else var for var in solution]


def random_search(clauses, initial_solutions, n_vars):
    all_scores = []

    for i in range(10):
        iterations = 0
        scores = []
        solution = initial_solutions[i]

        while iterations < max_iterations:
            iterations += 1
            score = evaluate_all(clauses, solution)
            scores.append(score)
            solution = disturb(solution)
        print(np.array(scores).max())
        all_scores.append(scores)

    all_scores = np.array(all_scores)
    best_scores = all_scores.max(axis=1)

    convergence = pd.DataFrame(all_scores.mean(axis=0),
                               np.arange(max_iterations),
                               columns=['clausulas'])

    plot_convergence(convergence, 'random_search')
    results["random_search"] = best_scores

    return best_scores.mean(), best_scores.std()


def simmulated_annealing(clauses, initial_solutions, n_vars):
    all_scores = []

    for i in range(10):
        solution = initial_solutions[i]
        score = evaluate_all(clauses, solution)
        temperature = t0
        iterations = 0
        scores = []

        while iterations < max_iterations:
            new_solution = disturb(solution)
            new_score = evaluate_all(clauses, new_solution)
            delta = score - new_score  # Equivalent to E(new_solution) - E(solution)
            if delta <= 0 or random() < np.exp(-delta/temperature):
                solution = new_solution
                score = new_score
            iterations += 1
            scores.append(score)
            temperature = next_temperature(iterations)
        print(np.array(scores).max())
        all_scores.append(scores)

    all_scores = np.array(all_scores)
    best_scores = all_scores.max(axis=1)

    convergence = pd.DataFrame(all_scores.mean(axis=0),
                               np.arange(max_iterations),
                               columns=['clausulas'])

    plot_convergence(convergence, 'simmulated_annealing')

    results["simmulated_annealing"] = best_scores

    return best_scores.mean(), best_scores.std()


def plot_convergence(df, name):
    df.index.name = 'Iterações'

    fig, ax = plt.subplots()
    fig.set_size_inches(30, 8.27)
    sns.lineplot(x=df.index, y="clausulas", data=df, lw=.4, ax=ax, estimator=None)
    sns.despine()
    fig.savefig('{}_{}_convergence.png'.format(instance, name))

    fig, ax = plt.subplots()
    fig.set_size_inches(30, 8.27)
    sns.lineplot(x=df.index, y="clausulas", data=df.rolling(1000).mean(), ax=ax)
    sns.despine()
    fig.savefig('{}_{}_convergence_agg.png'.format(instance, name))


filename = "{}.cnf".format(instance)
with open(filename, 'r') as file:
    all_lines = file.readlines()

clauses = []
for line in all_lines:
    if line.startswith('p'):
        _, _, n_vars, _ = line.split()
        n_vars = int(n_vars)

    elif line.startswith('%'):
        break

    elif not line.startswith('c'):
        v1, v2, v3, _ = line.split()
        clauses.append([to_tuple(v1), to_tuple(v2), to_tuple(v3)])

initial_solutions = [initial_solution(n_vars) for _ in range(10)]

mean, std = simmulated_annealing(clauses, initial_solutions, n_vars)
print("SA: {} +- {}".format(mean, std))
mean, std = random_search(clauses, initial_solutions, n_vars)
print("RS: {} +- {}".format(mean, std))

fig, ax = plt.subplots()
sns.boxplot(data=results, orient='v', ax=ax)
sns.despine()
fig.savefig('{}_boxblot.png'.format(instance))
