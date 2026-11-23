# Simple iteration (Jacobi) method for A*x = b.

def solve(A, b, tolerance=1e-8, max_iterations=10000):
    n = len(b)
    x = [0.0] * n

    for iteration in range(1, max_iterations + 1):
        new_x = [0.0] * n
        for i in range(n):
            if A[i][i] == 0:
                raise ValueError("Zero diagonal element")
            total = 0.0
            for j in range(n):
                if i != j:
                    total += A[i][j] * x[j]
            new_x[i] = (b[i] - total) / A[i][i]

        difference = max(abs(new_x[i] - x[i]) for i in range(n))
        x = new_x
        if difference < tolerance:
            return x, iteration

    raise RuntimeError("Jacobi did not converge")


if __name__ == "__main__":
    A = [[4.0, 1.0, 1.0], [1.0, 5.0, 1.0], [1.0, 1.0, 6.0]]
    b = [6.0, 7.0, 8.0]
    solution, iterations = solve(A, b)
    print("Solution:", solution)
    print("Iterations:", iterations)
