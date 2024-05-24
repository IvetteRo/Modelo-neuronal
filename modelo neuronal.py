#Ingeniería del conocimiento 1:00-2:00 Equipo 'Los Inge´s 2'
#Integrantes: González Escobedo, Rodríguez Tiscareño, Flores Ojeda, Ortega Villalobos
import random

# Definir la función de activación (función escalón)
def step_function(x):
    return 1 if x >= 0 else 0

# Entrenar el perceptrón
def train_perceptron(X, y, learning_rate, epochs):
    # Inicializar pesos y sesgo
    num_features = len(X[0])
    weights = [random.uniform(-1, 1) for _ in range(num_features)]
    bias = random.uniform(-1, 1)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        total_error = 0
        for inputs, target in zip(X, y):
            # Calcular la sumatoria (entrada ponderada)
            summation = sum(w * i for w, i in zip(weights, inputs)) + bias
            # Aplicar la función de activación
            output = step_function(summation)
            # Calcular el error
            error = target - output
            total_error += abs(error)
            # Actualizar los pesos y el sesgo si hay un error
            if error != 0:
                weights = [w + learning_rate * error * i for w, i in zip(weights, inputs)]
                bias += learning_rate * error
            print(f"  Inputs: {inputs}, Target: {target}, Output: {output}, Error: {error}")
            print(f"  Updated weights: {weights}, Updated bias: {bias}")
        print(f"Total error: {total_error}\n")
        # Si no hay error total, detener el entrenamiento
        if total_error == 0:
            break

    return weights, bias

# Generar datos de entrada y etiquetas para la compuerta AND
def generate_and_data(num_samples):
    X = [[random.randint(0, 1), random.randint(0, 1)] for _ in range(num_samples)]
    y = [1 if all(features) else 0 for features in X]  # Etiquetas para la compuerta AND
    return X, y

# Generar datos de entrada y etiquetas para la compuerta OR
def generate_or_data(num_samples):
    X = [[random.randint(0, 1), random.randint(0, 1)] for _ in range(num_samples)]
    y = [1 if any(features) else 0 for features in X]  # Etiquetas para la compuerta OR
    return X, y

# Generar datos de entrada y etiquetas para la compuerta XOR
def generate_xor_data(num_samples):
    X = [[random.randint(0, 1), random.randint(0, 1)] for _ in range(num_samples)]
    y = [1 if x[0] != x[1] else 0 for x in X]  # Etiquetas para la compuerta XOR
    return X, y

# Parámetros de entrenamiento
learning_rate = 0.1
epochs = 10

# Entrenar y probar el perceptrón para la compuerta AND
print("Entrenamiento para la compuerta AND:")
num_samples = 10  # Número de muestras
X_and, y_and = generate_and_data(num_samples)
weights_and, bias_and = train_perceptron(X_and, y_and, learning_rate, epochs)

# Resultados
print("\nResultados para la compuerta AND:")
print(f"Pesos finales: {weights_and}")
print(f"Sesgo final: {bias_and}")

# Prueba con los datos de entrada y crear la matriz de resultados
print("\nMatriz de datos de entrada y salidas predichas para la compuerta AND:")
print("Inputs | Predicted Output | y = w0*x0 + w1*x1 + bias")
print("---------------------------------------------------------")
for inputs in X_and:
    summation = sum(w * i for w, i in zip(weights_and, inputs)) + bias_and
    output = step_function(summation)
    inputs_str = ' '.join(map(str, inputs))
    print(f"{inputs_str} -> {output} | y = {summation:.2f}")

# Entrenar y probar el perceptrón para la compuerta OR
print("\nEntrenamiento para la compuerta OR:")
X_or, y_or = generate_or_data(num_samples)
weights_or, bias_or = train_perceptron(X_or, y_or, learning_rate, epochs)

# Resultados
print("\nResultados para la compuerta OR:")
print(f"Pesos finales: {weights_or}")
print(f"Sesgo final: {bias_or}")

# Prueba con los datos de entrada y crear la matriz de resultados
print("\nMatriz de datos de entrada y salidas predichas para la compuerta OR:")
print("Inputs | Predicted Output | y = w0*x0 + w1*x1 + bias")
print("---------------------------------------------------------")
for inputs in X_or:
    summation = sum(w * i for w, i in zip(weights_or, inputs)) + bias_or
    output = step_function(summation)
    inputs_str = ' '.join(map(str, inputs))
    print(f"{inputs_str} -> {output} | y = {summation:.2f}")

# Entrenar y probar el perceptrón para la compuerta XOR
print("\nEntrenamiento para la compuerta XOR:")
X_xor, y_xor = generate_xor_data(num_samples)
weights_xor, bias_xor = train_perceptron(X_xor, y_xor, learning_rate, epochs)

# Resultados
print("\nResultados para la compuerta XOR:")
print(f"Pesos finales: {weights_xor}")
print(f"Sesgo final: {bias_xor}")

# Prueba con los datos de entrada y crear la matriz de resultados
print("\nMatriz de datos de entrada y salidas predichas para la compuerta XOR:")
print("Inputs | Predicted Output | y = w0*x0 + w1*x1 + bias")
print("---------------------------------------------------------")
for inputs in X_xor:
    summation = sum(w * i for w, i in zip(weights_xor, inputs)) + bias_xor
    output = step_function(summation)
    inputs_str = ' '.join(map(str, inputs))
    print(f"{inputs_str} -> {output} | y = {summation:.2f}")