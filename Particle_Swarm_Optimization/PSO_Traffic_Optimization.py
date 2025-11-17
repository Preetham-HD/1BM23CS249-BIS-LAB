import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Setup
# -----------------------------
CYCLE_TIME = 120

# Weighted Fitness Functions
def fitness_two_weighted(x, weights=[0.7, 0.3]):
    """2-way: Weighted imbalance penalty (heavier traffic on NS)."""
    t_ns, t_ew = x
    ideal = np.array([CYCLE_TIME * 0.7, CYCLE_TIME * 0.3])
    wait = np.sum(weights * np.abs(x - ideal) ** 1.5)
    return wait

def fitness_four_weighted(x, weights=[0.4, 0.3, 0.2, 0.1]):
    """4-way: Weighted imbalance penalty (heavier traffic on NS straight)."""
    ideal = np.array([50, 40, 20, 10]) # based on demand
    wait = np.sum(weights * np.abs(x - ideal) ** 1.5)
    return wait


# -----------------------------
# Generic PSO Implementation
# -----------------------------
def PSO(fitness_func, dim, num_particles=20, max_iter=100,
        w=0.7, c1=1.5, c2=1.5, min_time=8):

    # Initialize random particles
    particles = np.random.rand(num_particles, dim) * CYCLE_TIME
    particles = np.array([p / sum(p) * CYCLE_TIME for p in particles])
    velocities = np.random.randn(num_particles, dim)

    # Initialize bests
    pbest_positions = particles.copy()
    pbest_scores = np.array([fitness_func(p) for p in particles])
    gbest_index = np.argmin(pbest_scores)
    gbest_position = pbest_positions[gbest_index].copy()
    gbest_score = pbest_scores[gbest_index]

    convergence = []
    best_positions_history = [] # Store best positions at each iteration

    for _ in range(max_iter):
        for i in range(num_particles):
            score = fitness_func(particles[i])

            if score < pbest_scores[i]:
                pbest_scores[i] = score
                pbest_positions[i] = particles[i].copy()

            if score < gbest_score:
                gbest_score = score
                gbest_position = particles[i].copy()

        for i in range(num_particles):
            r1, r2 = np.random.rand(), np.random.rand()
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (pbest_positions[i] - particles[i])
                + c2 * r2 * (gbest_position - particles[i])
            )
            particles[i] += velocities[i]

            # Normalize to cycle time
            total = np.sum(particles[i])
            particles[i] = particles[i] / total * CYCLE_TIME

            # Clip min green time
            particles[i] = np.clip(particles[i], min_time, CYCLE_TIME - min_time)
            total = np.sum(particles[i])
            particles[i] = particles[i] / total * CYCLE_TIME

        convergence.append(gbest_score)
        best_positions_history.append(gbest_position.copy())

    return gbest_position, gbest_score, convergence, best_positions_history


# -----------------------------
# Run PSO for 2-way intersection (weighted)
# -----------------------------
best_two, score_two, conv_two, history_two = PSO(fitness_two_weighted, dim=2)
print("=== 2-Way Weighted Traffic Optimization ===")
print("Optimal Signal Timings:", best_two)
print("Best Fitness (waiting cost):", score_two)
print("\nSignal Timings History (2-way):")
for i, pos in enumerate(history_two):
    print(f"Iteration {i+1}: {pos}")


# -----------------------------
# Run PSO for 4-way intersection (weighted)
# -----------------------------
best_four, score_four, conv_four, history_four = PSO(fitness_four_weighted, dim=4)
print("\n=== 4-Way Weighted Traffic Optimization ===")
print("Optimal Signal Timings:", best_four)
print("Best Fitness (waiting cost):", score_four)
print("\nSignal Timings History (4-way):")
for i, pos in enumerate(history_four):
    print(f"Iteration {i+1}: {pos}")


# -----------------------------
# Plot convergence
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(conv_two, label="2-way Weighted")
plt.plot(conv_four, label="4-way Weighted")
plt.xlabel("Iteration")
plt.ylabel("Best Fitness")
plt.title("PSO Convergence for Weighted Traffic Optimization")
plt.legend()
plt.grid(True)
plt.show()


# output:
#  === 2-Way Weighted Traffic Optimization ===
# Optimal Signal Timings: [83.99999998 36.00000002]
# Best Fitness (waiting cost): 3.283958671775337e-12

# Signal Timings History (2-way):
# Iteration 1: [83.62231644 36.37768356]
# Iteration 2: [84.29570561 35.70429439]
# Iteration 3: [84.29570561 35.70429439]
# Iteration 4: [84.29570561 35.70429439]
# Iteration 5: [83.99810763 36.00189237]
# Iteration 6: [83.99810763 36.00189237]
# Iteration 7: [83.99810763 36.00189237]
# Iteration 8: [83.99810763 36.00189237]
# Iteration 9: [83.99810763 36.00189237]
# Iteration 10: [83.99810763 36.00189237]
# Iteration 11: [83.99810763 36.00189237]
# Iteration 12: [83.99810763 36.00189237]
# Iteration 13: [83.99810763 36.00189237]
# Iteration 14: [83.99810763 36.00189237]
# Iteration 15: [83.99810763 36.00189237]
# Iteration 16: [83.99810763 36.00189237]
# Iteration 17: [83.99810763 36.00189237]
# Iteration 18: [83.99810763 36.00189237]
# Iteration 19: [83.99810763 36.00189237]
# Iteration 20: [83.99810763 36.00189237]
# Iteration 21: [83.99810763 36.00189237]
# Iteration 22: [83.99810763 36.00189237]
# Iteration 23: [83.99810763 36.00189237]
# Iteration 24: [83.99810763 36.00189237]
# Iteration 25: [83.99810763 36.00189237]
# Iteration 26: [83.99810763 36.00189237]
# Iteration 27: [83.99810763 36.00189237]
# Iteration 28: [83.99810763 36.00189237]
# Iteration 29: [83.99810763 36.00189237]
# Iteration 30: [84.0012872 35.9987128]
# Iteration 31: [84.0012872 35.9987128]
# Iteration 32: [84.0012872 35.9987128]
# Iteration 33: [83.99927264 36.00072736]
# Iteration 34: [83.99983126 36.00016874]
# Iteration 35: [83.99983126 36.00016874]
# Iteration 36: [83.99991276 36.00008724]
# Iteration 37: [83.99991276 36.00008724]
# Iteration 38: [83.99991276 36.00008724]
# Iteration 39: [84.00005636 35.99994364]
# Iteration 40: [84.00005636 35.99994364]
# Iteration 41: [84.00005061 35.99994939]
# Iteration 42: [83.99996515 36.00003485]
# Iteration 43: [83.99996515 36.00003485]
# Iteration 44: [83.99996515 36.00003485]
# Iteration 45: [84.00001159 35.99998841]
# Iteration 46: [84.00001159 35.99998841]
# Iteration 47: [83.99999402 36.00000598]
# Iteration 48: [83.99999402 36.00000598]
# Iteration 49: [83.99999402 36.00000598]
# Iteration 50: [83.99999402 36.00000598]
# Iteration 51: [84.00000304 35.99999696]
# Iteration 52: [84.00000304 35.99999696]
# Iteration 53: [84.00000304 35.99999696]
# Iteration 54: [84.00000304 35.99999696]
# Iteration 55: [84.00000304 35.99999696]
# Iteration 56: [84.00000304 35.99999696]
# Iteration 57: [84.00000304 35.99999696]
# Iteration 58: [84.00000304 35.99999696]
# Iteration 59: [84.00000304 35.99999696]
# Iteration 60: [84.00000304 35.99999696]
# Iteration 61: [84.00000304 35.99999696]
# Iteration 62: [84.00000304 35.99999696]
# Iteration 63: [84.00000304 35.99999696]
# Iteration 64: [84.00000304 35.99999696]
# Iteration 65: [83.99999835 36.00000165]
# Iteration 66: [83.99999835 36.00000165]
# Iteration 67: [83.99999835 36.00000165]
# Iteration 68: [83.99999835 36.00000165]
# Iteration 69: [83.99999879 36.00000121]
# Iteration 70: [83.99999879 36.00000121]
# Iteration 71: [84.00000113 35.99999887]
# Iteration 72: [84.00000113 35.99999887]
# Iteration 73: [84.00000113 35.99999887]
# Iteration 74: [84.00000113 35.99999887]
# Iteration 75: [84.00000113 35.99999887]
# Iteration 76: [84.00000113 35.99999887]
# Iteration 77: [84.00000113 35.99999887]
# Iteration 78: [84.00000109 35.99999891]
# Iteration 79: [84.00000033 35.99999967]
# Iteration 80: [84.00000033 35.99999967]
# Iteration 81: [84.00000033 35.99999967]
# Iteration 82: [84.00000033 35.99999967]
# Iteration 83: [84.00000033 35.99999967]
# Iteration 84: [84.00000033 35.99999967]
# Iteration 85: [84.00000033 35.99999967]
# Iteration 86: [84.00000033 35.99999967]
# Iteration 87: [84.00000013 35.99999987]
# Iteration 88: [83.99999998 36.00000002]
# Iteration 89: [83.99999998 36.00000002]
# Iteration 90: [83.99999998 36.00000002]
# Iteration 91: [83.99999998 36.00000002]
# Iteration 92: [83.99999998 36.00000002]
# Iteration 93: [83.99999998 36.00000002]
# Iteration 94: [83.99999998 36.00000002]
# Iteration 95: [83.99999998 36.00000002]
# Iteration 96: [83.99999998 36.00000002]
# Iteration 97: [83.99999998 36.00000002]
# Iteration 98: [83.99999998 36.00000002]
# Iteration 99: [83.99999998 36.00000002]
# Iteration 100: [83.99999998 36.00000002]

# === 4-Way Weighted Traffic Optimization ===
# Optimal Signal Timings: [50.29508191 40.52458834 21.18032975  8.        ]
# Best Fitness (waiting cost): 0.7174142934795764

# Signal Timings History (4-way):
# Iteration 1: [61.14243554 36.69923745 14.63432296  7.52400405]
# Iteration 2: [54.18901115 37.94324612 20.00630887  7.86143386]
# Iteration 3: [54.18901115 37.94324612 20.00630887  7.86143386]
# Iteration 4: [50.48549381 39.26050861 22.4885998   7.76539778]
# Iteration 5: [49.99209774 40.17138236 22.47034958  7.36617033]
# Iteration 6: [50.82743101 39.9117896  21.40556083  7.85521856]
# Iteration 7: [50.61691182 40.40829411 21.08857586  7.88621821]
# Iteration 8: [50.61691182 40.40829411 21.08857586  7.88621821]
# Iteration 9: [50.50891487 40.57007846 21.01793792  7.90306876]
# Iteration 10: [50.30942872 40.60639561 21.10849578  7.97567988]
# Iteration 11: [50.30942872 40.60639561 21.10849578  7.97567988]
# Iteration 12: [50.30942872 40.60639561 21.10849578  7.97567988]
# Iteration 13: [50.30942872 40.60639561 21.10849578  7.97567988]
# Iteration 14: [50.30942872 40.60639561 21.10849578  7.97567988]
# Iteration 15: [50.30942872 40.60639561 21.10849578  7.97567988]
# Iteration 16: [50.30403214 40.61999784 21.08959267  7.98637735]
# Iteration 17: [50.30403214 40.61999784 21.08959267  7.98637735]
# Iteration 18: [50.3043029  40.44437143 21.25846374  7.99286193]
# Iteration 19: [50.31780122 40.53648115 21.15194868  7.99376895]
# Iteration 20: [50.30495499 40.6019303  21.09682993  7.99628479]
# Iteration 21: [50.30495499 40.6019303  21.09682993  7.99628479]
# Iteration 22: [50.30495499 40.6019303  21.09682993  7.99628479]
# Iteration 23: [50.31901931 40.53987182 21.14293709  7.99817178]
# Iteration 24: [50.32184723 40.48693261 21.19250334  7.99871682]
# Iteration 25: [50.32184723 40.48693261 21.19250334  7.99871682]
# Iteration 26: [50.32184723 40.48693261 21.19250334  7.99871682]
# Iteration 27: [50.29935487 40.50639194 21.19492112  7.99933206]
# Iteration 28: [50.29935487 40.50639194 21.19492112  7.99933206]
# Iteration 29: [50.29908626 40.51739932 21.18376996  7.99974446]
# Iteration 30: [50.28578597 40.53578922 21.17860495  7.99981986]
# Iteration 31: [50.28578597 40.53578922 21.17860495  7.99981986]
# Iteration 32: [50.28853406 40.53189081 21.17967065  7.99990447]
# Iteration 33: [50.29697204 40.52014898 21.18294632  7.99993265]
# Iteration 34: [50.29697204 40.52014898 21.18294632  7.99993265]
# Iteration 35: [50.29454224 40.52349114 21.18200248  7.99996414]
# Iteration 36: [50.29454224 40.52349114 21.18200248  7.99996414]
# Iteration 37: [50.29354088 40.52486448 21.18161344  7.99998121]
# Iteration 38: [50.29354088 40.52486448 21.18161344  7.99998121]
# Iteration 39: [50.29167728 40.52744306 21.18088976  7.9999899 ]
# Iteration 40: [50.29167728 40.52744306 21.18088976  7.9999899 ]
# Iteration 41: [50.29109752 40.52824319 21.18066459  7.9999947 ]
# Iteration 42: [50.29326318 40.52523515 21.1815054   7.99999627]
# Iteration 43: [50.29477914 40.52312952 21.18209396  7.99999739]
# Iteration 44: [50.29477914 40.52312952 21.18209396  7.99999739]
# Iteration 45: [50.29504967 40.52275271 21.18219898  7.99999864]
# Iteration 46: [50.29449619 40.52352069 21.18198408  7.99999905]
# Iteration 47: [50.29410875 40.52405826 21.18183365  7.99999933]
# Iteration 48: [50.29383755 40.52443457 21.18172835  7.99999953]
# Iteration 49: [50.29383755 40.52443457 21.18172835  7.99999953]
# Iteration 50: [50.29382233 40.52445547 21.18172244  7.99999976]
# Iteration 51: [50.29394456 40.52428571 21.1817699   7.99999983]
# Iteration 52: [50.29403013 40.52416688 21.18180312  7.99999988]
# Iteration 53: [50.29409002 40.52408369 21.18182637  7.99999992]
# Iteration 54: [50.29413195 40.52402546 21.18184265  7.99999994]
# Iteration 55: [50.29498821 40.52495253 21.18005942  7.99999984]
# Iteration 56: [50.29479607 40.52467508 21.18052889  7.99999997]
# Iteration 57: [50.29479607 40.52467508 21.18052889  7.99999997]
# Iteration 58: [50.29479607 40.52467508 21.18052889  7.99999997]
# Iteration 59: [50.29479607 40.52467508 21.18052889  7.99999997]
# Iteration 60: [50.29487364 40.52479009 21.18033629  7.99999998]
# Iteration 61: [50.29497593 40.5248643  21.18015978  7.99999999]
# Iteration 62: [50.29492646 40.52461957 21.18045399  7.99999999]
# Iteration 63: [50.29492646 40.52461957 21.18045399  7.99999999]
# Iteration 64: [50.29492813 40.5247059  21.18036598  7.99999999]
# Iteration 65: [50.2949607  40.52474617 21.18029314  8.        ]
# Iteration 66: [50.2949607  40.52474617 21.18029314  8.        ]
# Iteration 67: [50.2949449  40.52471272 21.18034238  8.        ]
# Iteration 68: [50.2949449  40.52471272 21.18034238  8.        ]
# Iteration 69: [50.2949449  40.52471272 21.18034238  8.        ]
# Iteration 70: [50.29495285 40.52451948 21.18052767  8.        ]
# Iteration 71: [50.29494814 40.52464727 21.1804046   8.        ]
# Iteration 72: [50.29494814 40.52464727 21.1804046   8.        ]
# Iteration 73: [50.29494814 40.52464727 21.1804046   8.        ]
# Iteration 74: [50.29494814 40.52464727 21.1804046   8.        ]
# Iteration 75: [50.29494814 40.52464727 21.1804046   8.        ]
# Iteration 76: [50.29495073 40.52461366 21.18043561  8.        ]
# Iteration 77: [50.29495073 40.52461366 21.18043561  8.        ]
# Iteration 78: [50.29496533 40.52466739 21.18036728  8.        ]
# Iteration 79: [50.29502899 40.52451952 21.1804515   8.        ]
# Iteration 80: [50.29505333 40.52461114 21.18033553  8.        ]
# Iteration 81: [50.29505333 40.52461114 21.18033553  8.        ]
# Iteration 82: [50.29508088 40.52457612 21.180343    8.        ]
# Iteration 83: [50.29508088 40.52457612 21.180343    8.        ]
# Iteration 84: [50.29508088 40.52457612 21.180343    8.        ]
# Iteration 85: [50.29508088 40.52457612 21.180343    8.        ]
# Iteration 86: [50.2950826  40.52457752 21.18033988  8.        ]
# Iteration 87: [50.2950826  40.52457752 21.18033988  8.        ]
# Iteration 88: [50.29508727 40.52458134 21.18033139  8.        ]
# Iteration 89: [50.29508727 40.52458134 21.18033139  8.        ]
# Iteration 90: [50.29508608 40.52458037 21.18033355  8.        ]
# Iteration 91: [50.29508608 40.52458037 21.18033355  8.        ]
# Iteration 92: [50.29508632 40.52458057 21.18033312  8.        ]
# Iteration 93: [50.29508632 40.52458057 21.18033312  8.        ]
# Iteration 94: [50.2950828  40.52458485 21.18033234  8.        ]
# Iteration 95: [50.2950828  40.52458485 21.18033234  8.        ]
# Iteration 96: [50.2950828  40.52458485 21.18033234  8.        ]
# Iteration 97: [50.29508555 40.52458678 21.18032767  8.        ]
# Iteration 98: [50.29508341 40.52458769 21.1803289   8.        ]
# Iteration 99: [50.29508191 40.52458834 21.18032975  8.        ]
# Iteration 100: [50.29508191 40.52458834 21.18032975  8.        ]
