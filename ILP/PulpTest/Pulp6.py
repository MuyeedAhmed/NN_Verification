from pulp import LpProblem, LpVariable, LpMaximize, LpStatus, lpSum

# Fixed X (10x2), W1 (2x2), W2 (2x1), and y (10x1)
X = [
    [2, 3], [4, 1], [5, -2], [-3, 6], [7, -4],
    [0, 2], [-1, 5], [3, -3], [2, -5], [4, 0]
]
W1 = [[1, -1], [2, 3]]  # First-layer weights (2x2)
W2 = [[0.5], [-1.5]]    # Second-layer weights (2x1)
y = [1, 0, 1, 1, 0, 1, 0, 1, 0, 1]  # Fixed binary labels

# Small epsilon for strict inequality
epsilon = 1e-6  

# Define the LP problem (MAXIMIZING b)
lp = LpProblem(name="Find-Maximal-B", sense=LpMaximize)

# Define b1 as a 1x2 continuous variable array
b1 = [LpVariable(f"b1_{i}", cat="Continuous") for i in range(2)]
b2 = LpVariable("b2", cat="Continuous")

# Define variables for ReLU output A1 (10x2)
A1 = [[LpVariable(f"A1_{i}_{j}", lowBound=0, cat="Continuous") for j in range(2)] for i in range(10)]

# Compute Z1 = X W1 + b1 and enforce ReLU constraints
for i in range(10):
    for j in range(2):
        Z1_ij = X[i][0] * W1[0][j] + X[i][1] * W1[1][j] + b1[j]

        # ReLU constraints: A1_ij = max(0, Z1_ij)
        lp += A1[i][j] >= Z1_ij, f"ReLU_{i}_{j}_lower"
        lp += A1[i][j] >= 0, f"ReLU_{i}_{j}_nonneg"

# Compute Z2 = A1 W2 + b2 and enforce classification constraints
for i in range(10):
    Z2_i = lpSum(A1[i][j] * W2[j][0] for j in range(2)) + b2

    # Enforce classification rule: y_pred[i] == y[i]
    if y[i] == 1:
        lp += Z2_i >= epsilon, f"Z2_{i}_positive"
    else:
        lp += Z2_i <= -epsilon, f"Z2_{i}_negative"

# Objective: Maximize each b1[i] and b2
lp += lpSum(b1) + b2, "Maximize_b"

# Solve the LP
lp.solve()

# Print results
if LpStatus[lp.status] == "Optimal":
    b1_values = [b1[i].varValue for i in range(2)]
    b2_value = b2.varValue
    print("Optimal solution found (Maximization):")
    print(f"b1: {b1_values}")
    print(f"b2: {b2_value}")
else:
    print("No feasible solution found.")
