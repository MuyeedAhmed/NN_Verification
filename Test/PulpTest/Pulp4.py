from pulp import LpProblem, LpVariable, LpMinimize, LpStatus, lpSum

# Fixed values for X
X = [2, 4, 1, 5, -3, 6, -2, 7, 3, -4]  # Example fixed array

# Small epsilon to enforce strict inequality
epsilon = 1e-6

# Define the LP problem (we minimize T)
lp = LpProblem(name="Find-Minimal-T", sense=LpMinimize)

# Define A as a continuous variable array
A = [LpVariable(f"A_{i}", cat="Continuous") for i in range(10)]

# Define T as a continuous variable
T = LpVariable("T", cat="Continuous")

# Constraint: Sum of element-wise multiplication must be exactly T
lp += lpSum(A[i] + (A[i] * X[i]) for i in range(10)) == T, "Sum_Constraint"

# Constraints: T must be strictly greater than all values in X and A
for i in range(10):
    lp += T >= A[i] + 1, f"T_strictly_greater_than_A_{i}"
    lp += T >= X[i] + 1, f"T_strictly_greater_than_X_{i}"

# Objective: Minimize T
lp += T

# Solve the LP
lp.solve()

# Print results
if LpStatus[lp.status] == "Optimal":
    print("Optimal solution found:")
    A_values = [A[i].varValue for i in range(10)]
    T_value = T.varValue
    print(f"A: {A_values}")
    print(f"T: {T_value}")
    print("Verification: ", sum(A_values[i] * X[i] for i in range(10)), "==", T_value)
    print(f"Checking strict inequality: T > max(X) and T > max(A) -> {T_value > max(X) and T_value > max(A_values)}")
else:
    print("No feasible solution found.")
