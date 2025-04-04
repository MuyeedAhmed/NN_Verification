from pulp import LpProblem, LpVariable, LpConstraint, LpStatus, lpSum

X = [2, 4, 1, 5, -3, 6, -2, 7, 3, -4]
T = 2

lp = LpProblem(name="Find-Noise-A")

A = [LpVariable(f"A_{i}", cat="Continuous") for i in range(10)]

lp += lpSum(A[i]*2 * X[i] for i in range(10)) == T, "Sum_Constraint"

lp.solve()

if LpStatus[lp.status] == "Optimal":
    print("Found an optimal solution for A:")
    A_values = [A[i].varValue for i in range(10)]
    print(A_values)
    print("Verification: ", sum(A_values[i] * X[i] for i in range(10)), "==", T)
else:
    print("No feasible solution found.")
