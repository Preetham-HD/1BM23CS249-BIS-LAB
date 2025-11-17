import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# =========================================================
# 1. Load MNIST subset
# =========================================================
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.astype(np.float32)[:1000] # use 1000 samples for speed
y = mnist.target.astype(int)[:1000]

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =========================================================
# 2. GA + Neural Network
# =========================================================

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def forward_pass(weights, X, input_size, hidden_size, output_size):
    # unpack weights
    ih_size = input_size * hidden_size
    ho_size = hidden_size * output_size
    w1 = weights[:ih_size].reshape(input_size, hidden_size)
    w2 = weights[ih_size:ih_size+ho_size].reshape(hidden_size, output_size)
    b1 = weights[ih_size+ho_size:ih_size+ho_size+hidden_size]
    b2 = weights[ih_size+ho_size+hidden_size:]
    # forward
    h = np.tanh(X @ w1 + b1)
    o = np.array([softmax(h[i] @ w2 + b2) for i in range(X.shape[0])])
    return o

def accuracy_nn(weights, X, y, input_size, hidden_size, output_size):
    preds = forward_pass(weights, X, input_size, hidden_size, output_size)
    return np.mean(np.argmax(preds, axis=1) == y)

def evolve_nn(X_train, y_train, X_test, y_test,
              input_size=784, hidden_size=32, output_size=10,
              pop_size=30, generations=50, mutation_rate=0.1):

    genome_length = input_size*hidden_size + hidden_size*output_size + hidden_size + output_size
    population = [np.random.randn(genome_length) for _ in range(pop_size)]

    best_accs, avg_accs = [], []

    for gen in range(generations):
        scores = [accuracy_nn(ind, X_train, y_train, input_size, hidden_size, output_size) for ind in population]
        best_idx = np.argmax(scores)
        best_acc, best_ind = scores[best_idx], population[best_idx]
        print(f"[GA+NN] Gen {gen+1}, Best Acc = {best_acc:.3f}, Avg Acc = {np.mean(scores):.3f}")
        best_accs.append(best_acc)
        avg_accs.append(np.mean(scores))

        # next population
        new_pop = [best_ind] # elitism
        while len(new_pop) < pop_size:
            parents = random.sample(population, 2)
            cross_point = random.randint(0, genome_length-1)
            child = np.concatenate([parents[0][:cross_point], parents[1][cross_point:]])
            if random.random() < mutation_rate:
                child[random.randint(0, genome_length-1)] += np.random.randn()
            new_pop.append(child)
        population = new_pop

    final_test_acc = accuracy_nn(best_ind, X_test, y_test, input_size, hidden_size, output_size)
    print("[GA+NN] Final Test Accuracy:", final_test_acc)

    return best_accs, avg_accs

# =========================================================
# 3. Mini GEP (expression trees)
# =========================================================

def add(a, b): return a + b
def sub(a, b): return a - b
def mul(a, b): return a * b
def safe_div(a, b): return a / b if b != 0 else a

FUNCTIONS = [add, sub, mul, safe_div]

def random_expression(num_features, depth=3):
    if depth == 0 or random.random() < 0.3:
        return ("x", random.randint(0, num_features-1))
    func = random.choice(FUNCTIONS)
    return (func, random_expression(num_features, depth-1), random_expression(num_features, depth-1))

def eval_expr(expr, sample):
    if expr[0] == "x":
        return sample[expr[1]]
    func, left, right = expr
    return func(eval_expr(left, sample), eval_expr(right, sample))

def fitness(expr, X, y):
    preds = []
    for row in X:
        val = eval_expr(expr, row)
        preds.append(int(val) % 10)
    return accuracy_score(y, preds)

def mutate(expr, num_features):
    if random.random() < 0.2:
        return random_expression(num_features)
    if expr[0] == "x":
        return ("x", random.randint(0, num_features-1))
    func, left, right = expr
    return (func, mutate(left, num_features), mutate(right, num_features))

def crossover(e1, e2):
    return random.choice([e1, e2]) # simple crossover

# ---- Pretty print expression ----
def expr_to_str(expr):
    if expr[0] == "x":
        return f"x{expr[1]}"
    func, left, right = expr
    func_name = {add:"+", sub:"-", mul:"*", safe_div:"/"}[func]
    return f"({expr_to_str(left)} {func_name} {expr_to_str(right)})"

def evolve_gep(X_train, y_train, X_test, y_test, pop_size=20, generations=50):
    population = [random_expression(X_train.shape[1]) for _ in range(pop_size)]
    best_accs = []
    best_expr = None

    for gen in range(generations):
        scored = [(fitness(expr, X_train, y_train), expr) for expr in population]
        scored.sort(key=lambda x: x[0], reverse=True)
        best_fit, best_expr = scored[0]
        best_accs.append(best_fit)
        print(f"[GEP] Gen {gen+1}, Best Acc = {best_fit:.3f}")

        # next population
        new_pop = [best_expr]
        while len(new_pop) < pop_size:
            parent = random.choice(scored[:5])[1]
            if random.random() < 0.5:
                child = mutate(parent, X_train.shape[1])
            else:
                mate = random.choice(scored[:5])[1]
                child = crossover(parent, mate)
            new_pop.append(child)
        population = new_pop

    final_test_acc = fitness(best_expr, X_test, y_test)
    print("[GEP] Final Test Accuracy:", final_test_acc)
    print("[GEP] Best Expression:", expr_to_str(best_expr))

    return best_accs

# =========================================================
# 4. Run both and compare
# =========================================================

ga_best, ga_avg = evolve_nn(X_train, y_train, X_test, y_test, generations=200)
gep_best = evolve_gep(X_train, y_train, X_test, y_test, generations=200)

# Plot comparison
plt.plot(ga_best, label="GA+NN Best")
plt.plot(ga_avg, label="GA+NN Avg")
plt.plot(gep_best, label="GEP Best")
plt.xlabel("Generation")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# output:
#  [GA+NN] Gen 1, Best Acc = 0.172, Avg Acc = 0.099
# [GA+NN] Gen 2, Best Acc = 0.172, Avg Acc = 0.101
# [GA+NN] Gen 3, Best Acc = 0.172, Avg Acc = 0.107
# [GA+NN] Gen 4, Best Acc = 0.172, Avg Acc = 0.110
# [GA+NN] Gen 5, Best Acc = 0.186, Avg Acc = 0.105
# [GA+NN] Gen 6, Best Acc = 0.196, Avg Acc = 0.109
# [GA+NN] Gen 7, Best Acc = 0.196, Avg Acc = 0.108
# [GA+NN] Gen 8, Best Acc = 0.196, Avg Acc = 0.108
# [GA+NN] Gen 9, Best Acc = 0.196, Avg Acc = 0.122
# [GA+NN] Gen 10, Best Acc = 0.196, Avg Acc = 0.114
# [GA+NN] Gen 11, Best Acc = 0.200, Avg Acc = 0.112
# [GA+NN] Gen 12, Best Acc = 0.200, Avg Acc = 0.120
# [GA+NN] Gen 13, Best Acc = 0.200, Avg Acc = 0.124
# [GA+NN] Gen 14, Best Acc = 0.201, Avg Acc = 0.122
# [GA+NN] Gen 15, Best Acc = 0.201, Avg Acc = 0.120
# [GA+NN] Gen 16, Best Acc = 0.209, Avg Acc = 0.126
# [GA+NN] Gen 17, Best Acc = 0.209, Avg Acc = 0.127
# [GA+NN] Gen 18, Best Acc = 0.223, Avg Acc = 0.133
# [GA+NN] Gen 19, Best Acc = 0.223, Avg Acc = 0.128
# [GA+NN] Gen 20, Best Acc = 0.223, Avg Acc = 0.139
# [GA+NN] Gen 21, Best Acc = 0.229, Avg Acc = 0.140
# [GA+NN] Gen 22, Best Acc = 0.229, Avg Acc = 0.143
# [GA+NN] Gen 23, Best Acc = 0.229, Avg Acc = 0.139
# [GA+NN] Gen 24, Best Acc = 0.229, Avg Acc = 0.145
# [GA+NN] Gen 25, Best Acc = 0.229, Avg Acc = 0.145
# [GA+NN] Gen 26, Best Acc = 0.229, Avg Acc = 0.150
# [GA+NN] Gen 27, Best Acc = 0.229, Avg Acc = 0.152
# [GA+NN] Gen 28, Best Acc = 0.229, Avg Acc = 0.148
# [GA+NN] Gen 29, Best Acc = 0.229, Avg Acc = 0.154
# [GA+NN] Gen 30, Best Acc = 0.229, Avg Acc = 0.152
# [GA+NN] Gen 31, Best Acc = 0.229, Avg Acc = 0.149
# [GA+NN] Gen 32, Best Acc = 0.229, Avg Acc = 0.144
# [GA+NN] Gen 33, Best Acc = 0.229, Avg Acc = 0.143
# [GA+NN] Gen 34, Best Acc = 0.229, Avg Acc = 0.139
# [GA+NN] Gen 35, Best Acc = 0.229, Avg Acc = 0.131
# [GA+NN] Gen 36, Best Acc = 0.229, Avg Acc = 0.136
# [GA+NN] Gen 37, Best Acc = 0.229, Avg Acc = 0.130
# [GA+NN] Gen 38, Best Acc = 0.229, Avg Acc = 0.138
# [GA+NN] Gen 39, Best Acc = 0.229, Avg Acc = 0.141
# [GA+NN] Gen 40, Best Acc = 0.229, Avg Acc = 0.144
# [GA+NN] Gen 41, Best Acc = 0.229, Avg Acc = 0.147
# [GA+NN] Gen 42, Best Acc = 0.229, Avg Acc = 0.138
# [GA+NN] Gen 43, Best Acc = 0.229, Avg Acc = 0.139
# [GA+NN] Gen 44, Best Acc = 0.229, Avg Acc = 0.141
# [GA+NN] Gen 45, Best Acc = 0.229, Avg Acc = 0.130
# [GA+NN] Gen 46, Best Acc = 0.229, Avg Acc = 0.134
# [GA+NN] Gen 47, Best Acc = 0.229, Avg Acc = 0.136
# [GA+NN] Gen 48, Best Acc = 0.229, Avg Acc = 0.141
# [GA+NN] Gen 49, Best Acc = 0.229, Avg Acc = 0.145
# [GA+NN] Gen 50, Best Acc = 0.229, Avg Acc = 0.155
# [GA+NN] Gen 51, Best Acc = 0.229, Avg Acc = 0.157
# [GA+NN] Gen 52, Best Acc = 0.229, Avg Acc = 0.166
# [GA+NN] Gen 53, Best Acc = 0.229, Avg Acc = 0.171
# [GA+NN] Gen 54, Best Acc = 0.229, Avg Acc = 0.162
# [GA+NN] Gen 55, Best Acc = 0.229, Avg Acc = 0.164
# [GA+NN] Gen 56, Best Acc = 0.229, Avg Acc = 0.168
# [GA+NN] Gen 57, Best Acc = 0.229, Avg Acc = 0.177
# [GA+NN] Gen 58, Best Acc = 0.229, Avg Acc = 0.175
# [GA+NN] Gen 59, Best Acc = 0.235, Avg Acc = 0.181
# [GA+NN] Gen 60, Best Acc = 0.236, Avg Acc = 0.182
# [GA+NN] Gen 61, Best Acc = 0.236, Avg Acc = 0.194
# [GA+NN] Gen 62, Best Acc = 0.236, Avg Acc = 0.195
# [GA+NN] Gen 63, Best Acc = 0.236, Avg Acc = 0.201
# [GA+NN] Gen 64, Best Acc = 0.236, Avg Acc = 0.206
# [GA+NN] Gen 65, Best Acc = 0.236, Avg Acc = 0.202
# [GA+NN] Gen 66, Best Acc = 0.236, Avg Acc = 0.206
# [GA+NN] Gen 67, Best Acc = 0.236, Avg Acc = 0.204
# [GA+NN] Gen 68, Best Acc = 0.236, Avg Acc = 0.205
# [GA+NN] Gen 69, Best Acc = 0.236, Avg Acc = 0.205
# [GA+NN] Gen 70, Best Acc = 0.236, Avg Acc = 0.202
# [GA+NN] Gen 71, Best Acc = 0.241, Avg Acc = 0.206
# [GA+NN] Gen 72, Best Acc = 0.241, Avg Acc = 0.204
# [GA+NN] Gen 73, Best Acc = 0.241, Avg Acc = 0.198
# [GA+NN] Gen 74, Best Acc = 0.241, Avg Acc = 0.199
# [GA+NN] Gen 75, Best Acc = 0.241, Avg Acc = 0.199
# [GA+NN] Gen 76, Best Acc = 0.241, Avg Acc = 0.207
# [GA+NN] Gen 77, Best Acc = 0.241, Avg Acc = 0.212
# [GA+NN] Gen 78, Best Acc = 0.241, Avg Acc = 0.215
# [GA+NN] Gen 79, Best Acc = 0.247, Avg Acc = 0.211
# [GA+NN] Gen 80, Best Acc = 0.247, Avg Acc = 0.216
# [GA+NN] Gen 81, Best Acc = 0.247, Avg Acc = 0.224
# [GA+NN] Gen 82, Best Acc = 0.247, Avg Acc = 0.221
# [GA+NN] Gen 83, Best Acc = 0.247, Avg Acc = 0.219
# [GA+NN] Gen 84, Best Acc = 0.247, Avg Acc = 0.218
# [GA+NN] Gen 85, Best Acc = 0.247, Avg Acc = 0.220
# [GA+NN] Gen 86, Best Acc = 0.247, Avg Acc = 0.218
# [GA+NN] Gen 87, Best Acc = 0.247, Avg Acc = 0.216
# [GA+NN] Gen 88, Best Acc = 0.247, Avg Acc = 0.219
# [GA+NN] Gen 89, Best Acc = 0.247, Avg Acc = 0.220
# [GA+NN] Gen 90, Best Acc = 0.247, Avg Acc = 0.217
# [GA+NN] Gen 91, Best Acc = 0.247, Avg Acc = 0.212
# [GA+NN] Gen 92, Best Acc = 0.247, Avg Acc = 0.211
# [GA+NN] Gen 93, Best Acc = 0.247, Avg Acc = 0.211
# [GA+NN] Gen 94, Best Acc = 0.247, Avg Acc = 0.215
# [GA+NN] Gen 95, Best Acc = 0.249, Avg Acc = 0.213
# [GA+NN] Gen 96, Best Acc = 0.249, Avg Acc = 0.209
# [GA+NN] Gen 97, Best Acc = 0.249, Avg Acc = 0.213
# [GA+NN] Gen 98, Best Acc = 0.249, Avg Acc = 0.217
# [GA+NN] Gen 99, Best Acc = 0.249, Avg Acc = 0.214
# [GA+NN] Gen 100, Best Acc = 0.249, Avg Acc = 0.215
# [GA+NN] Gen 101, Best Acc = 0.249, Avg Acc = 0.216
# [GA+NN] Gen 102, Best Acc = 0.249, Avg Acc = 0.215
# [GA+NN] Gen 103, Best Acc = 0.249, Avg Acc = 0.218
# [GA+NN] Gen 104, Best Acc = 0.249, Avg Acc = 0.219
# [GA+NN] Gen 105, Best Acc = 0.249, Avg Acc = 0.220
# [GA+NN] Gen 106, Best Acc = 0.249, Avg Acc = 0.221
# [GA+NN] Gen 107, Best Acc = 0.249, Avg Acc = 0.220
# [GA+NN] Gen 108, Best Acc = 0.249, Avg Acc = 0.222
# [GA+NN] Gen 109, Best Acc = 0.249, Avg Acc = 0.221
# [GA+NN] Gen 110, Best Acc = 0.249, Avg Acc = 0.217
# [GA+NN] Gen 111, Best Acc = 0.249, Avg Acc = 0.219
# [GA+NN] Gen 112, Best Acc = 0.249, Avg Acc = 0.220
# [GA+NN] Gen 113, Best Acc = 0.249, Avg Acc = 0.221
# [GA+NN] Gen 114, Best Acc = 0.249, Avg Acc = 0.225
# [GA+NN] Gen 115, Best Acc = 0.249, Avg Acc = 0.225
# [GA+NN] Gen 116, Best Acc = 0.249, Avg Acc = 0.231
# [GA+NN] Gen 117, Best Acc = 0.249, Avg Acc = 0.236
# [GA+NN] Gen 118, Best Acc = 0.249, Avg Acc = 0.231
# [GA+NN] Gen 119, Best Acc = 0.249, Avg Acc = 0.231
# [GA+NN] Gen 120, Best Acc = 0.249, Avg Acc = 0.228
# [GA+NN] Gen 121, Best Acc = 0.249, Avg Acc = 0.234
# [GA+NN] Gen 122, Best Acc = 0.249, Avg Acc = 0.237
# [GA+NN] Gen 123, Best Acc = 0.249, Avg Acc = 0.236
# [GA+NN] Gen 124, Best Acc = 0.249, Avg Acc = 0.234
# [GA+NN] Gen 125, Best Acc = 0.249, Avg Acc = 0.230
# [GA+NN] Gen 126, Best Acc = 0.249, Avg Acc = 0.227
# [GA+NN] Gen 127, Best Acc = 0.249, Avg Acc = 0.225
# [GA+NN] Gen 128, Best Acc = 0.249, Avg Acc = 0.225
# [GA+NN] Gen 129, Best Acc = 0.249, Avg Acc = 0.224
# [GA+NN] Gen 130, Best Acc = 0.249, Avg Acc = 0.227
# [GA+NN] Gen 131, Best Acc = 0.249, Avg Acc = 0.229
# [GA+NN] Gen 132, Best Acc = 0.249, Avg Acc = 0.227
# [GA+NN] Gen 133, Best Acc = 0.249, Avg Acc = 0.227
# [GA+NN] Gen 134, Best Acc = 0.249, Avg Acc = 0.230
# [GA+NN] Gen 135, Best Acc = 0.249, Avg Acc = 0.234
# [GA+NN] Gen 136, Best Acc = 0.249, Avg Acc = 0.240
# [GA+NN] Gen 137, Best Acc = 0.250, Avg Acc = 0.240
# [GA+NN] Gen 138, Best Acc = 0.250, Avg Acc = 0.243
# [GA+NN] Gen 139, Best Acc = 0.251, Avg Acc = 0.245
# [GA+NN] Gen 140, Best Acc = 0.251, Avg Acc = 0.244
# [GA+NN] Gen 141, Best Acc = 0.251, Avg Acc = 0.243
# [GA+NN] Gen 142, Best Acc = 0.251, Avg Acc = 0.245
# [GA+NN] Gen 143, Best Acc = 0.251, Avg Acc = 0.244
# [GA+NN] Gen 144, Best Acc = 0.251, Avg Acc = 0.244
# [GA+NN] Gen 145, Best Acc = 0.251, Avg Acc = 0.243
# [GA+NN] Gen 146, Best Acc = 0.251, Avg Acc = 0.243
# [GA+NN] Gen 147, Best Acc = 0.251, Avg Acc = 0.245
# [GA+NN] Gen 148, Best Acc = 0.251, Avg Acc = 0.244
# [GA+NN] Gen 149, Best Acc = 0.251, Avg Acc = 0.243
# [GA+NN] Gen 150, Best Acc = 0.251, Avg Acc = 0.245
# [GA+NN] Gen 151, Best Acc = 0.251, Avg Acc = 0.245
# [GA+NN] Gen 152, Best Acc = 0.251, Avg Acc = 0.245
# [GA+NN] Gen 153, Best Acc = 0.251, Avg Acc = 0.247
# [GA+NN] Gen 154, Best Acc = 0.251, Avg Acc = 0.248
# [GA+NN] Gen 155, Best Acc = 0.251, Avg Acc = 0.249
# [GA+NN] Gen 156, Best Acc = 0.251, Avg Acc = 0.249
# [GA+NN] Gen 157, Best Acc = 0.251, Avg Acc = 0.249
# [GA+NN] Gen 158, Best Acc = 0.251, Avg Acc = 0.249
# [GA+NN] Gen 159, Best Acc = 0.251, Avg Acc = 0.248
# [GA+NN] Gen 160, Best Acc = 0.251, Avg Acc = 0.248
# [GA+NN] Gen 161, Best Acc = 0.251, Avg Acc = 0.248
# [GA+NN] Gen 162, Best Acc = 0.253, Avg Acc = 0.249
# [GA+NN] Gen 163, Best Acc = 0.253, Avg Acc = 0.248
# [GA+NN] Gen 164, Best Acc = 0.253, Avg Acc = 0.249
# [GA+NN] Gen 165, Best Acc = 0.253, Avg Acc = 0.249
# [GA+NN] Gen 166, Best Acc = 0.253, Avg Acc = 0.249
# [GA+NN] Gen 167, Best Acc = 0.253, Avg Acc = 0.249
# [GA+NN] Gen 168, Best Acc = 0.253, Avg Acc = 0.248
# [GA+NN] Gen 169, Best Acc = 0.253, Avg Acc = 0.248
# [GA+NN] Gen 170, Best Acc = 0.253, Avg Acc = 0.248
# [GA+NN] Gen 171, Best Acc = 0.253, Avg Acc = 0.248
# [GA+NN] Gen 172, Best Acc = 0.253, Avg Acc = 0.249
# [GA+NN] Gen 173, Best Acc = 0.253, Avg Acc = 0.248
# [GA+NN] Gen 174, Best Acc = 0.253, Avg Acc = 0.249
# [GA+NN] Gen 175, Best Acc = 0.253, Avg Acc = 0.250
# [GA+NN] Gen 176, Best Acc = 0.253, Avg Acc = 0.250
# [GA+NN] Gen 177, Best Acc = 0.253, Avg Acc = 0.250
# [GA+NN] Gen 178, Best Acc = 0.253, Avg Acc = 0.250
# [GA+NN] Gen 179, Best Acc = 0.253, Avg Acc = 0.250
# [GA+NN] Gen 180, Best Acc = 0.253, Avg Acc = 0.250
# [GA+NN] Gen 181, Best Acc = 0.253, Avg Acc = 0.250
# [GA+NN] Gen 182, Best Acc = 0.253, Avg Acc = 0.250
# [GA+NN] Gen 183, Best Acc = 0.253, Avg Acc = 0.250
# [GA+NN] Gen 184, Best Acc = 0.253, Avg Acc = 0.251
# [GA+NN] Gen 185, Best Acc = 0.253, Avg Acc = 0.250
# [GA+NN] Gen 186, Best Acc = 0.253, Avg Acc = 0.250
# [GA+NN] Gen 187, Best Acc = 0.253, Avg Acc = 0.250
# [GA+NN] Gen 188, Best Acc = 0.253, Avg Acc = 0.251
# [GA+NN] Gen 189, Best Acc = 0.253, Avg Acc = 0.251
# [GA+NN] Gen 190, Best Acc = 0.253, Avg Acc = 0.251
# [GA+NN] Gen 191, Best Acc = 0.253, Avg Acc = 0.251
# [GA+NN] Gen 192, Best Acc = 0.253, Avg Acc = 0.251
# [GA+NN] Gen 193, Best Acc = 0.253, Avg Acc = 0.251
# [GA+NN] Gen 194, Best Acc = 0.253, Avg Acc = 0.252
# [GA+NN] Gen 195, Best Acc = 0.253, Avg Acc = 0.251
# [GA+NN] Gen 196, Best Acc = 0.253, Avg Acc = 0.251
# [GA+NN] Gen 197, Best Acc = 0.254, Avg Acc = 0.251
# [GA+NN] Gen 198, Best Acc = 0.254, Avg Acc = 0.251
# [GA+NN] Gen 199, Best Acc = 0.254, Avg Acc = 0.251
# [GA+NN] Gen 200, Best Acc = 0.254, Avg Acc = 0.252
# [GA+NN] Final Test Accuracy: 0.2
# [GEP] Gen 1, Best Acc = 0.140
# [GEP] Gen 2, Best Acc = 0.140
# [GEP] Gen 3, Best Acc = 0.140
# [GEP] Gen 4, Best Acc = 0.146
# [GEP] Gen 5, Best Acc = 0.146
# [GEP] Gen 6, Best Acc = 0.158
# [GEP] Gen 7, Best Acc = 0.158
# [GEP] Gen 8, Best Acc = 0.158
# [GEP] Gen 9, Best Acc = 0.175
# [GEP] Gen 10, Best Acc = 0.175
# [GEP] Gen 11, Best Acc = 0.175
# [GEP] Gen 12, Best Acc = 0.175
# [GEP] Gen 13, Best Acc = 0.176
# [GEP] Gen 14, Best Acc = 0.176
# [GEP] Gen 15, Best Acc = 0.176
# [GEP] Gen 16, Best Acc = 0.176
# [GEP] Gen 17, Best Acc = 0.176
# [GEP] Gen 18, Best Acc = 0.204
# [GEP] Gen 19, Best Acc = 0.204
# [GEP] Gen 20, Best Acc = 0.204
# [GEP] Gen 21, Best Acc = 0.204
# [GEP] Gen 22, Best Acc = 0.204
# [GEP] Gen 23, Best Acc = 0.204
# [GEP] Gen 24, Best Acc = 0.204
# [GEP] Gen 25, Best Acc = 0.204
# [GEP] Gen 26, Best Acc = 0.204
# [GEP] Gen 27, Best Acc = 0.204
# [GEP] Gen 28, Best Acc = 0.204
# [GEP] Gen 29, Best Acc = 0.204
# [GEP] Gen 30, Best Acc = 0.204
# [GEP] Gen 31, Best Acc = 0.204
# [GEP] Gen 32, Best Acc = 0.204
# [GEP] Gen 33, Best Acc = 0.204
# [GEP] Gen 34, Best Acc = 0.204
# [GEP] Gen 35, Best Acc = 0.204
# [GEP] Gen 36, Best Acc = 0.204
# [GEP] Gen 37, Best Acc = 0.204
# [GEP] Gen 38, Best Acc = 0.204
# [GEP] Gen 39, Best Acc = 0.204
# [GEP] Gen 40, Best Acc = 0.204
# [GEP] Gen 41, Best Acc = 0.204
# [GEP] Gen 42, Best Acc = 0.204
# [GEP] Gen 43, Best Acc = 0.204
# [GEP] Gen 44, Best Acc = 0.204
# [GEP] Gen 45, Best Acc = 0.204
# [GEP] Gen 46, Best Acc = 0.204
# [GEP] Gen 47, Best Acc = 0.204
# [GEP] Gen 48, Best Acc = 0.204
# [GEP] Gen 49, Best Acc = 0.204
# [GEP] Gen 50, Best Acc = 0.204
# [GEP] Gen 51, Best Acc = 0.204
# [GEP] Gen 52, Best Acc = 0.204
# [GEP] Gen 53, Best Acc = 0.204
# [GEP] Gen 54, Best Acc = 0.204
# [GEP] Gen 55, Best Acc = 0.204
# [GEP] Gen 56, Best Acc = 0.204
# [GEP] Gen 57, Best Acc = 0.204
# [GEP] Gen 58, Best Acc = 0.204
# [GEP] Gen 59, Best Acc = 0.230
# [GEP] Gen 60, Best Acc = 0.230
# [GEP] Gen 61, Best Acc = 0.230
# [GEP] Gen 62, Best Acc = 0.230
# [GEP] Gen 63, Best Acc = 0.230
# [GEP] Gen 64, Best Acc = 0.230
# [GEP] Gen 65, Best Acc = 0.230
# [GEP] Gen 66, Best Acc = 0.230
# [GEP] Gen 67, Best Acc = 0.230
# [GEP] Gen 68, Best Acc = 0.230
# [GEP] Gen 69, Best Acc = 0.230
# [GEP] Gen 70, Best Acc = 0.230
# [GEP] Gen 71, Best Acc = 0.230
# [GEP] Gen 72, Best Acc = 0.230
# [GEP] Gen 73, Best Acc = 0.230
# [GEP] Gen 74, Best Acc = 0.230
# [GEP] Gen 75, Best Acc = 0.230
# [GEP] Gen 76, Best Acc = 0.230
# [GEP] Gen 77, Best Acc = 0.230
# [GEP] Gen 78, Best Acc = 0.230
# [GEP] Gen 79, Best Acc = 0.230
# [GEP] Gen 80, Best Acc = 0.230
# [GEP] Gen 81, Best Acc = 0.230
# [GEP] Gen 82, Best Acc = 0.230
# [GEP] Gen 83, Best Acc = 0.230
# [GEP] Gen 84, Best Acc = 0.230
# [GEP] Gen 85, Best Acc = 0.230
# [GEP] Gen 86, Best Acc = 0.230
# [GEP] Gen 87, Best Acc = 0.230
# [GEP] Gen 88, Best Acc = 0.230
# [GEP] Gen 89, Best Acc = 0.230
# [GEP] Gen 90, Best Acc = 0.230
# [GEP] Gen 91, Best Acc = 0.230
# [GEP] Gen 92, Best Acc = 0.230
# [GEP] Gen 93, Best Acc = 0.230
# [GEP] Gen 94, Best Acc = 0.230
# [GEP] Gen 95, Best Acc = 0.230
# [GEP] Gen 96, Best Acc = 0.230
# [GEP] Gen 97, Best Acc = 0.230
# [GEP] Gen 98, Best Acc = 0.230
# [GEP] Gen 99, Best Acc = 0.230
# [GEP] Gen 100, Best Acc = 0.230
# [GEP] Gen 101, Best Acc = 0.230
# [GEP] Gen 102, Best Acc = 0.230
# [GEP] Gen 103, Best Acc = 0.230
# [GEP] Gen 104, Best Acc = 0.230
# [GEP] Gen 105, Best Acc = 0.230
# [GEP] Gen 106, Best Acc = 0.230
# [GEP] Gen 107, Best Acc = 0.230
# [GEP] Gen 108, Best Acc = 0.230
# [GEP] Gen 109, Best Acc = 0.230
# [GEP] Gen 110, Best Acc = 0.230
# [GEP] Gen 111, Best Acc = 0.230
# [GEP] Gen 112, Best Acc = 0.230
# [GEP] Gen 113, Best Acc = 0.230
# [GEP] Gen 114, Best Acc = 0.230
# [GEP] Gen 115, Best Acc = 0.230
# [GEP] Gen 116, Best Acc = 0.230
# [GEP] Gen 117, Best Acc = 0.230
# [GEP] Gen 118, Best Acc = 0.230
# [GEP] Gen 119, Best Acc = 0.230
# [GEP] Gen 120, Best Acc = 0.230
# [GEP] Gen 121, Best Acc = 0.230
# [GEP] Gen 122, Best Acc = 0.230
# [GEP] Gen 123, Best Acc = 0.230
# [GEP] Gen 124, Best Acc = 0.230
# [GEP] Gen 125, Best Acc = 0.230
# [GEP] Gen 126, Best Acc = 0.230
# [GEP] Gen 127, Best Acc = 0.230
# [GEP] Gen 128, Best Acc = 0.230
# [GEP] Gen 129, Best Acc = 0.230
# [GEP] Gen 130, Best Acc = 0.230
# [GEP] Gen 131, Best Acc = 0.230
# [GEP] Gen 132, Best Acc = 0.230
# [GEP] Gen 133, Best Acc = 0.230
# [GEP] Gen 134, Best Acc = 0.230
# [GEP] Gen 135, Best Acc = 0.230
# [GEP] Gen 136, Best Acc = 0.230
# [GEP] Gen 137, Best Acc = 0.230
# [GEP] Gen 138, Best Acc = 0.230
# [GEP] Gen 139, Best Acc = 0.230
# [GEP] Gen 140, Best Acc = 0.230
# [GEP] Gen 141, Best Acc = 0.230
# [GEP] Gen 142, Best Acc = 0.230
# [GEP] Gen 143, Best Acc = 0.230
# [GEP] Gen 144, Best Acc = 0.230
# [GEP] Gen 145, Best Acc = 0.230
# [GEP] Gen 146, Best Acc = 0.230
# [GEP] Gen 147, Best Acc = 0.230
# [GEP] Gen 148, Best Acc = 0.230
# [GEP] Gen 149, Best Acc = 0.230
# [GEP] Gen 150, Best Acc = 0.230
# [GEP] Gen 151, Best Acc = 0.230
# [GEP] Gen 152, Best Acc = 0.230
# [GEP] Gen 153, Best Acc = 0.230
# [GEP] Gen 154, Best Acc = 0.230
# [GEP] Gen 155, Best Acc = 0.230
# [GEP] Gen 156, Best Acc = 0.230
# [GEP] Gen 157, Best Acc = 0.230
# [GEP] Gen 158, Best Acc = 0.230
# [GEP] Gen 159, Best Acc = 0.230
# [GEP] Gen 160, Best Acc = 0.230
# [GEP] Gen 161, Best Acc = 0.230
# [GEP] Gen 162, Best Acc = 0.230
# [GEP] Gen 163, Best Acc = 0.230
# [GEP] Gen 164, Best Acc = 0.230
# [GEP] Gen 165, Best Acc = 0.230
# [GEP] Gen 166, Best Acc = 0.230
# [GEP] Gen 167, Best Acc = 0.230
# [GEP] Gen 168, Best Acc = 0.230
# [GEP] Gen 169, Best Acc = 0.230
# [GEP] Gen 170, Best Acc = 0.230
# [GEP] Gen 171, Best Acc = 0.230
# [GEP] Gen 172, Best Acc = 0.230
# [GEP] Gen 173, Best Acc = 0.230
# [GEP] Gen 174, Best Acc = 0.230
# [GEP] Gen 175, Best Acc = 0.230
# [GEP] Gen 176, Best Acc = 0.230
# [GEP] Gen 177, Best Acc = 0.230
# [GEP] Gen 178, Best Acc = 0.230
# [GEP] Gen 179, Best Acc = 0.230
# [GEP] Gen 180, Best Acc = 0.230
# [GEP] Gen 181, Best Acc = 0.230
# [GEP] Gen 182, Best Acc = 0.230
# [GEP] Gen 183, Best Acc = 0.230
# [GEP] Gen 184, Best Acc = 0.230
# [GEP] Gen 185, Best Acc = 0.230
# [GEP] Gen 186, Best Acc = 0.230
# [GEP] Gen 187, Best Acc = 0.230
# [GEP] Gen 188, Best Acc = 0.230
# [GEP] Gen 189, Best Acc = 0.230
# [GEP] Gen 190, Best Acc = 0.230
# [GEP] Gen 191, Best Acc = 0.230
# [GEP] Gen 192, Best Acc = 0.230
# [GEP] Gen 193, Best Acc = 0.230
# [GEP] Gen 194, Best Acc = 0.230
# [GEP] Gen 195, Best Acc = 0.230
# [GEP] Gen 196, Best Acc = 0.230
# [GEP] Gen 197, Best Acc = 0.230
# [GEP] Gen 198, Best Acc = 0.230
# [GEP] Gen 199, Best Acc = 0.230
# [GEP] Gen 200, Best Acc = 0.230
# [GEP] Final Test Accuracy: 0.195
# [GEP] Best Expression: (x765 - ((x39 * (((((x386 / x220) - (x489 / x322)) / ((x277 - x570) - (x278 * x458))) - ((x387 * (x324 - x187)) - ((x373 / x266) / x617))) + (((x175 * x203) * (x211 * x599)) * ((x416 / x591) + (x23 / x421))))) - ((((x262 / x635) * (x69 / x778)) - x56) / x119)))

