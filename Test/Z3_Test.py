from z3 import *

def forward_pass_with_constraints(X, y, W1, b1, W2, b2):
    def relu(x):
        return If(x > 0, x, 0)

    def sigmoid_approx(x):
        return x / (1 + Abs(x))

    W1_offset = [[Real(f'W1_offset_{i}_{j}') for j in range(len(W1[0]))] for i in range(len(W1))]
    W2_offset = [[Real(f'W2_offset_{i}_{j}') for j in range(len(W2[0]))] for i in range(len(W2))]

    NewW1 = [[W1[i][j] + W1_offset[i][j] for j in range(len(W1[0]))] for i in range(len(W1))]
    NewW2 = [[W2[i][j] + W2_offset[i][j] for j in range(len(W2[0]))] for i in range(len(W2))]

    Z1 = [sum(X[i] * NewW1[i][j] for i in range(len(X))) + b1[j] for j in range(len(W1[0]))]
    A1 = [relu(Z1[j]) for j in range(len(Z1))]

    Z2 = [sum(A1[i] * NewW2[i][j] for i in range(len(A1))) + b2[j] for j in range(len(W2[0]))]
    A2 = [relu(Z2[j]) for j in range(len(Z2))]


    Z3 = sum(A2)
    y_prob = sigmoid_approx(Z3)
    y_predict = Real('Y_pred')
    y_predict = If(y_prob >= 0.5, 1, 0)

    return y_predict, W1_offset, W2_offset
    
    
input_size = 3
hidden_size1 = 5
hidden_size2 = 5

X = [0.5, -1.2, 0.8]
y = 1

W1 = [[0.2, -0.3, 0.5, -0.7, 0.1], [0.4, -0.2, 0.6, 0.3, -0.5], [-0.1, 0.8, -0.4, 0.7, 0.2]]
b1 = [0.1, -0.1, 0.2, 0.0, -0.2]

W2 = [[-0.3, 0.5, 0.7, -0.6, 0.2], [0.4, -0.8, 0.3, 0.1, -0.7], [0.2, -0.5, 0.6, 0.4, 0.8], [-0.2, 0.1, -0.7, 0.3, -0.5], [0.6, -0.3, 0.4, 0.7, -0.1]]
b2 = [0.2, -0.1, 0.3, 0.1, -0.4]

y_predict, W1_offset, W2_offset = forward_pass_with_constraints(X, y, W1, b1, W2, b2)

solver = Solver()

solver.add(y_predict == y)
print(solver.assertions())
if solver.check() == sat:
    model = solver.model()
    print("Solution found:")
    # for i in range(len(W1_offset)):
    #     for j in range(len(W1_offset[0])):
    #         print(f"W1_offset[{i}][{j}] =", model[W1_offset[i][j]])
    
    # for i in range(len(W2_offset)):
    #     for j in range(len(W2_offset[0])):
    #         print(f"W2_offset[{i}][{j}] =", model[W2_offset[i][j]])
    # print("y pred", model.eval(y_predict)) 
else:
    print("No solution found.")

