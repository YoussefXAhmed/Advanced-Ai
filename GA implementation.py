import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import random

columns = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach",
           "exang","oldpeak","slope","ca","thal","target"]
df = pd.read_csv("heart.csv", names=columns, na_values='?')
df = df.dropna()  # drop missing values

X = df.drop("target", axis=1)
y = df["target"].apply(lambda v: 1 if v > 0 else 0)  # Binary classification

X = pd.get_dummies(X, columns=["cp","restecg","slope","thal"], drop_first=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

POP_SIZE = 20
NUM_GENERATIONS = 50
CROSSOVER_PROB = 0.8
MUTATION_PROB = 0.1
num_features = X_train.shape[1]

def fitness(chromosome):
    selected_features = [i for i in range(num_features) if chromosome[i] == 1]
    if len(selected_features) == 0:
        return 0
    X_subset = X_train[:, selected_features]
    knn = KNeighborsClassifier(n_neighbors=5)
    score = cross_val_score(knn, X_subset, y_train, cv=5).mean()
    return score

population = [np.random.randint(0, 2, num_features) for _ in range(POP_SIZE)]
best_chromosome = None
best_fitness = 0

for generation in range(NUM_GENERATIONS):
    fitness_scores = [fitness(ind) for ind in population]
    
    max_fit_idx = np.argmax(fitness_scores)
    if fitness_scores[max_fit_idx] > best_fitness:
        best_fitness = fitness_scores[max_fit_idx]
        best_chromosome = population[max_fit_idx]
    
    total_fitness = sum(fitness_scores)
    if total_fitness == 0:
        probabilities = [1/POP_SIZE]*POP_SIZE
    else:
        probabilities = [f/total_fitness for f in fitness_scores]
    selected = np.random.choice(range(POP_SIZE), size=POP_SIZE, p=probabilities)
    selected_population = [population[i] for i in selected]
    
    next_population = []
    for i in range(0, POP_SIZE, 2):
        parent1 = selected_population[i]
        parent2 = selected_population[i+1 if i+1 < POP_SIZE else 0]
        if random.random() < CROSSOVER_PROB:
            point = random.randint(1, num_features-1)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
        else:
            child1, child2 = parent1.copy(), parent2.copy()
        next_population.extend([child1, child2])
    
    for ind in next_population:
        for j in range(num_features):
            if random.random() < MUTATION_PROB:
                ind[j] = 1 - ind[j]
    
    population = next_population

selected_features_indices = [i for i in range(num_features) if best_chromosome[i] == 1]
selected_features_names = X.columns[selected_features_indices].tolist()

print("GA Feature Selection Completed")
print("Selected Features:")
print(selected_features_names)
print(f"Best Fitness (KNN Accuracy): {best_fitness:.4f}")
