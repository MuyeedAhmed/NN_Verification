from pulp import LpProblem, LpVariable, LpMinimize, LpStatus, lpSum

X = [
    [2, 3], [4, 1], [5, -2], [-3, 6], [7, -4],
    [0, 2], [-1, 5], [3, -3], [2, -5], [4, 0]
]
W1 = [[1, -1], [2, 3]]
W2 = [[0.5], [-1.5]]
y = [1, 0, 1, 1, 0, 1, 0, 1, 0, 1]

epsilon = 1e-6  
# M = 100

lp = LpProblem(name="Find-Minimal-B", sense=LpMinimize)

b1 = [LpVariable(f"b1_{i}", cat="Continuous") for i in range(2)]
b2 = LpVariable("b2", cat="Continuous")

b1_abs = [LpVariable(f"b1_abs_{i}", lowBound=0, cat="Continuous") for i in range(2)]
b2_abs = LpVariable("b2_abs", lowBound=0, cat="Continuous")

for i in range(2):
    lp += b1_abs[i] >= b1[i], f"abs_b1_{i}_pos"
    lp += b1_abs[i] >= -b1[i], f"abs_b1_{i}_neg"

lp += b2_abs >= b2, "abs_b2_pos"
lp += b2_abs >= -b2, "abs_b2_neg"

A1 = [[LpVariable(f"A1_{i}_{j}", lowBound=0, cat="Continuous") for j in range(2)] for i in range(10)]

for i in range(10):
    for j in range(2):
        Z1_ij = X[i][0] * W1[0][j] + X[i][1] * W1[1][j] + b1[j]

        lp += A1[i][j] >= Z1_ij, f"ReLU_{i}_{j}_lower"
        lp += A1[i][j] >= 0, f"ReLU_{i}_{j}_nonneg"

for i in range(10):
    Z2_i = lpSum(A1[i][j] * W2[j][0] for j in range(2)) + b2

    # Enforce classification rule: y_pred[i] == y[i]
    if y[i] == 1:
        lp += Z2_i >= epsilon, f"Z2_{i}_positive"
    else:
        lp += Z2_i <= -epsilon, f"Z2_{i}_negative"


lp += lpSum(b1_abs) + b2_abs, "Minimize_b_abs"
# lp += (b1[0]**2 + b1[1]**2 + b2**2), "Minimize_b_squared"

lp += b1[0] >= 100

# lp += b1[0] <= 1
# lp += b1[0] >= -1
# lp += b1[1] <= 1
# lp += b1[1] >= -1
# lp += b2 <= 1
# lp += b2 >= -1


lp.solve()

# Print results
if LpStatus[lp.status] == "Optimal":
    b1_values = [b1[i].varValue for i in range(2)]
    b2_value = b2.varValue
    print("Optimal solution found:")
    print(f"b1: {b1_values}")
    print(f"b2: {b2_value}")
else:
    print("No feasible solution found.")
