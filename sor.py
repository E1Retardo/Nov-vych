# Successive Over-Relaxation (SOR) method for A*x = b.

def solve(A, b, omega=1.1, tolerance=1e-8, max_iterations=10000):
    if not 0 < omega < 2:
        raise ValueError("omega must be between 0 and 2")

    n = len(b)
    x = [0.0] * n

    for iteration in range(1, max_iterations + 1):
        previous_x = x.copy()
        for i in range(n):
            if A[i][i] == 0:
                raise ValueError("Zero diagonal element")
            total = 0.0
            for j in range(n):
                if i != j:
                    total += A[i][j] * x[j]
            seidel_value = (b[i] - total) / A[i][i]
            x[i] = (1 - omega) * previous_x[i] + omega * seidel_value

        difference = max(abs(x[i] - previous_x[i]) for i in range(n))
        if difference < tolerance:
            return x, iteration

    raise RuntimeError("SOR did not converge")


if __name__ == "__main__":
    A = [[4.0, 1.0, 1.0], [1.0, 5.0, 1.0], [1.0, 1.0, 6.0]]
    b = [6.0, 7.0, 8.0]
    solution, iterations = solve(A, b, omega=1.1)
    print("Solution:", solution)
    print("Iterations:", iterations)
