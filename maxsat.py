from random import randint, random
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

instance = 'uf250-01'
max_iterations = 250000
t0 = 100
tf = 0


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


def random_search(clauses, initial_solutions, n_vars):
    all_scores = []

    for i in range(10):
        iterations = 0
        scores = []

        while iterations < max_iterations:
            iterations += 1
            solution = initial_solution(n_vars)
            score = evaluate_all(clauses, solution)
            scores.append(score)

        all_scores.append(scores)

    all_scores = np.array(all_scores)
    df = pd.DataFrame(all_scores.mean(axis=0), np.arange(max_iterations), columns=['score'])
    plot_convergence(df, 'random_search')

    best_scores = all_scores.max(axis=1)

    return best_scores.mean(), best_scores.std()


def next_temperature(i):
    # return t0*(tf/t0)**(i/max_iterations)
    # return t0 - i*((t0-tf)/max_iterations)

    return (t0-tf)/(np.cosh(10*i/max_iterations)) + tf
    # return (t0-tf)/float(1 + np.exp(3*(i-max_iterations/2.0))) + tf
    # return t0 - i**(np.log(t0-tf)/np.log(max_iterations))

    # too greedy:
    # a = ((t0 - tf)*(max_iterations + 1))/max_iterations
    # b = t0 - a
    # return (a/(i+1) + b)

    #a = 1/(max_iterations**2)*np.log(t0/tf)
    # return t0*np.exp(-a*i**2)

    # return 0.5*(t0-tf)*(1-np.tanh(10*i/max_iterations - 5)) + tf

    # return 0.5*(t0-tf)*(1+np.cos(np.pi*i/max_iterations)) + tf


def simmulated_annealing(clauses, initial_solutions, n_vars):
    all_scores = []

    for i in range(10):
        new_solution = solution = initial_solutions[i]
        score = evaluate_all(clauses, solution)
        temperature = t0
        iterations = 0

        scores = []

        while iterations < max_iterations:
            disturbance = randint(0, len(solution)-1)
            new_solution[disturbance] = 1 - solution[disturbance]
            new_score = evaluate_all(clauses, new_solution)
            delta = (n_vars - new_score) - (n_vars - score)
            if delta <= 0 or random() < np.exp(-delta/temperature):
                solution = new_solution
                score = new_score
            iterations += 1
            # if (iterations % 1000 == 0):
            # print(temperature)
            scores.append(score)
            temperature = next_temperature(iterations)

        print(score)
        all_scores.append(scores)

    all_scores = np.array(all_scores)
    df = pd.DataFrame(all_scores.mean(axis=0), np.arange(max_iterations), columns=['score'])
    plot_convergence(df, 'simmulated_annealing')

    best_scores = all_scores.max(axis=1)

    return best_scores.mean(), best_scores.std()

    # return all_scores.mean(axis=0)[-1], all_scores.std(axis=0)[-1]


def plot_convergence(df, name):
    fig, ax = plt.subplots()
    fig.set_size_inches(30, 8.27)
    sns.lineplot(data=df, lw=.2, ax=ax, estimator=None)
    sns.despine()
    fig.savefig('{}_{}_convergence.png'.format(instance, name))

    fig, ax = plt.subplots()
    fig.set_size_inches(30, 8.27)
    sns.lineplot(data=df.rolling(1000).mean(), ax=ax)
    sns.despine()
    fig.savefig('{}_{}_convergence_agg.png'.format(instance, name))


filename = "{}.cnf".format(instance)
with open(filename, 'r') as file:
    all_lines = file.readlines()

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

initial_solutions = [initial_solution(n_vars) for _ in range(10)]

mean, std = simmulated_annealing(clauses, initial_solutions, n_vars)
print("SA: {} +- {}".format(mean, std))
mean, std = random_search(clauses, initial_solutions, n_vars)
print("RS: {} +- {}".format(mean, std))
