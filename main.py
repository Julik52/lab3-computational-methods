# Модель: Точка ринкової рівноваги для товарів-субститутів (5 семестр)
# Автор: Паршикова Юлія, група AI-231

import numpy as np
import matplotlib.pyplot as plt
import time

# --- 1. ПАРАМЕТРИ МОДЕЛІ ---
params = {
    'a0': 100, 'a1': 2, 'a2': 1,    # Ринок A (попит)
    'b0': 10, 'b1': 3,               # Ринок A (пропозиція)
    'c0': 80, 'c1': 1, 'c2': 0.5,   # Ринок Б (попит)
    'd0': 5, 'd1': 2                # Ринок Б (пропозиція)
}

def formulate_system(params):
    """
    Формує систему лінійних рівнянь A*X = B з параметрів моделі.
    X = [P_A, P_B]
    """
    print("--- 1. Формування системи A*X = B ---")

    # Рівняння 1: (a1+b1)*P_A - a2*P_B = a0-b0
    eq1_pa = params['a1'] + params['b1']
    eq1_pb = -params['a2']
    eq1_b = params['a0'] - params['b0']

    # Рівняння 2: -c2*P_A + (c1+d1)*P_B = c0-d0
    eq2_pa = -params['c2']
    eq2_pb = params['c1'] + params['d1']
    eq2_b = params['c0'] - params['d0']

    # Створення матриць
    A = np.array([
        [eq1_pa, eq1_pb],
        [eq2_pa, eq2_pb]
    ])

    B = np.array([eq1_b, eq2_b])

    print(f"Рівняння 1: {eq1_pa:.1f}*P_A + ({eq1_pb:.1f})*P_B = {eq1_b:.1f}")
    print(f"Рівняння 2: ({eq2_pa:.1f})*P_A + {eq2_pb:.1f}*P_B = {eq2_b:.1f}\n")
    print("Матриця A:\n", A)
    print("\nВектор B:\n", B)

    return A, B

def solve_direct(A, B):
    """
    Розв'язує систему A*X = B прямим аналітичним методом.
    """
    print("\n--- 2. Метод 1: Прямий розв'язок (np.linalg.solve) ---")

    start_time = time.perf_counter()

    # Головна обчислювальна функція
    try:
        solution = np.linalg.solve(A, B)
        end_time = time.perf_counter()

        print(f"Рішення знайдено.")
        print(f"  Рівноважна ціна P_A: {solution[0]:.10f}")
        print(f"  Рівноважна ціна P_B: {solution[1]:.10f}")
        print(f"  Час виконання: {(end_time - start_time):.10f} сек.")
        print("  Кількість ітерацій: 1 (прямий розв'язок)")
        return solution

    except np.linalg.LinAlgError:
        print("Помилка: Матриця A є сингулярною, розв'язку не існує або він не єдиний.")
        return None

def solve_gauss_seidel(A, B, epsilon=1e-6, max_iterations=100):
    """
    Розв'язує систему A*X = B ітераційним методом Гауса-Зейделя.
    """
    print("\n--- 3. Метод 2: Ітераційний (Метод Гауса-Зейделя) ---")

    start_time = time.perf_counter()

    n = len(B)
    X = np.zeros(n)

    for k in range(max_iterations):
        X_prev = np.copy(X)

        for i in range(n):
            s1 = np.dot(A[i, :i], X[:i])

            s2 = np.dot(A[i, i+1:], X_prev[i+1:])

            X[i] = (B[i] - s1 - s2) / A[i, i]

        # Перевірка умови збіжності
        error = np.linalg.norm(X - X_prev, ord=np.inf)
        if error < epsilon:
            break

    end_time = time.perf_counter()

    print(f"Збіжність досягнута.")
    print(f"  Рівноважна ціна P_A: {X[0]:.10f}")
    print(f"  Рівноважна ціна P_B: {X[1]:.10f}")
    print(f"  Час виконання: {(end_time - start_time):.10f} сек.")
    print(f"  Кількість ітерацій: {k+1}")
    print(f"  Досягнута точність (похибка): {error:.2e}")

    return X

def visualize_equilibrium(solution, A, B):
    """
    Будує графік, що ілюструє ринкову рівновагу.
    """
    print("\n--- 4. Візуалізація результатів ---")

    if solution is None:
        print("Візуалізація неможлива, оскільки розв'язок не знайдено.")
        return

    P_A_star, P_B_star = solution

    # Створюємо діапазон цін P_A для побудови графіків
    # Беремо ціни навколо точки рівноваги
    P_A_values = np.linspace(P_A_star - 10, P_A_star + 10, 100)

    # --- Перетворюємо рівняння A*X = B у P_B = f(P_A) ---

    # Рівняння 1: A[0,0]*P_A + A[0,1]*P_B = B[0]
    # P_B = (B[0] - A[0,0]*P_A) / A[0,1]
    P_B_line1 = (B[0] - A[0,0] * P_A_values) / A[0,1]

    # Рівняння 2: A[1,0]*P_A + A[1,1]*P_B = B[1]
    # P_B = (B[1] - A[1,0]*P_A) / A[1,1]
    P_B_line2 = (B[1] - A[1,0] * P_A_values) / A[1,1]

    # --- Побудова графіка ---

    plt.figure(figsize=(10, 6))
    plt.plot(P_A_values, P_B_line1, label=f"Ринок A (Рівняння 1: {A[0,0]:.1f}*PA + {A[0,1]:.1f}*PB = {B[0]:.1f})")
    plt.plot(P_A_values, P_B_line2, label=f"Ринок Б (Рівняння 2: {A[1,0]:.1f}*PA + {A[1,1]:.1f}*PB = {B[1]:.1f})")

    # Позначаємо точку рівноваги
    plt.plot(P_A_star, P_B_star, 'ro', markersize=10, label=f"Точка Рівноваги ({P_A_star:.2f}, {P_B_star:.2f})")

    plt.title("Ринкова рівновага для товарів-субститутів")
    plt.xlabel("Ціна Товару A ($P_A$)")
    plt.ylabel("Ціна Товару Б ($P_B$)")
    plt.legend()
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.show()

    print("Графік успішно побудовано.")

if __name__ == "__main__":
    A_matrix, B_vector = formulate_system(params)

    direct_sol = solve_direct(A_matrix, B_vector)
    gauss_sol = solve_gauss_seidel(A_matrix, B_vector)

    visualize_equilibrium(direct_sol, A_matrix, B_vector)

