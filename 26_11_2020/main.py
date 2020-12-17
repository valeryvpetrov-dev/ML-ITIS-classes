import random

FIRST_POPULATION_SIZE = 10
POPULATIONS_LIMIT = 200
REPRODUCTION_PAIRS = 20
PARAMETERS_NUMBER = 3  # x, y, z


def generate_random_param() -> int:
    return random.randint(-60, 60)


def calculate_equation(x: int, y: int, z: int) -> int:
    return 2 * x + 2 * y + 2 * z


def calculate_equation_dist(x: int, y: int, z: int) -> int:
    return abs(calculate_equation(x, y, z) - 60)


def check_equation_dist_for_goal(dist: int) -> bool:
    return dist == 0


def get_equation_solution(solutions: list, solutions_dists: list) -> list or None:
    for i in range(len(solutions_dists)):
        if solutions_dists[i] == 0:
            return solutions[i]
    return None


def calculate_survival_prob(dists: list) -> list:
    survival_prob = []
    for dist in dists:
        survival_prob.append(1 / dist)
    sum_survival_prob = sum(survival_prob)
    return list(map(lambda x: x / sum_survival_prob, survival_prob))


def build_reproduction_pairs(populations: list, population_survival_probs: list) -> list:
    reproduction_pairs = []
    for i in range(REPRODUCTION_PAIRS):
        first_parent = random.choices(populations, population_survival_probs)[0]
        second_parent = None
        while second_parent is None or second_parent == first_parent:
            second_parent = random.choices(populations, population_survival_probs)[0]
        reproduction_pairs.append((first_parent, second_parent))
    return reproduction_pairs


def crossover_genes(reproduction_pairs: list) -> list:
    new_population = []
    for reproduction_pair in reproduction_pairs:
        child = []
        first_parent_genes_number = random.randint(1, PARAMETERS_NUMBER)
        second_parent_genes_number = PARAMETERS_NUMBER - first_parent_genes_number
        crossover_genes_from_parent(child, reproduction_pair[0], first_parent_genes_number)
        crossover_genes_from_parent(child, reproduction_pair[1], second_parent_genes_number)
        new_population.append(child)
    return new_population


def crossover_genes_from_parent(child: list, parent: list, genes_number_to_crossover: int):
    parent_genes = parent.copy()
    for i in range(genes_number_to_crossover):
        first_parent_gene = parent_genes[random.randint(0, len(parent_genes) - 1)]
        parent_genes.remove(first_parent_gene)
        child.append(first_parent_gene)


def mutate_population(population: list):
    for member in population:
        mutation_param_i = random.randint(0, PARAMETERS_NUMBER - 1)
        member[mutation_param_i] = generate_random_param()


if __name__ == '__main__':
    """
    Solves Diophantine equation 2x + 2y + 2z = 60
    * assume what x, y, z are in range [-60; 60]
    """
    # create first population
    population = [
        [generate_random_param() for j in range(PARAMETERS_NUMBER)] for i in range(FIRST_POPULATION_SIZE)
    ]
    # while goal or population limit is not achieved
    populations_passed = 0
    solution = None
    previous_population_survival_prob = None
    while populations_passed != POPULATIONS_LIMIT or solution is not None:
        populations_passed += 1
        # calculate how population is close to goal
        population_dists = [calculate_equation_dist(*population[i]) for i in range(len(population))]
        # check if solution found
        solution = get_equation_solution(population, population_dists)
        if solution is not None:
            break
        # calculate survival probability
        population_survival_probs = calculate_survival_prob(population_dists)
        # build pairs for reproduction
        reproduction_pairs = build_reproduction_pairs(population, population_survival_probs)
        # create new population using parent genes crossover
        population = crossover_genes(reproduction_pairs)
        # indicate if mutation is needed
        current_population_survival_prob = sum(population_survival_probs)
        if previous_population_survival_prob is not None \
                and current_population_survival_prob < previous_population_survival_prob:
            mutate_population(population)
        previous_population_survival_prob = current_population_survival_prob
    print('Solution of 2x + 2y + 2z = 60 is found = ({}, {}, {})'.format(*solution))
    print('Populations passed = {}'.format(populations_passed))
