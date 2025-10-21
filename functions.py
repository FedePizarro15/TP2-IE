import math
import matplotlib.pyplot as plt
import numpy as np

# EJERCICIO 1

def frequency_character(file: str) -> dict[str, int]:
    frequencies = {}

    with open(file, 'r', encoding='utf-8') as f: # encoding para leer tildes o ñ
        content = f.read()

        for character in content:
            if character in frequencies:
                frequencies[character] += 1
            else:
                frequencies[character] = 1
    
    return frequencies
    
def probability_mass(file: str) -> dict[str, float]:
    frequencies = frequency_character(file)

    probability_dict = {}

    total_characters = sum(frequencies.values())

    for character in frequencies.keys():
        probability_dict[character] = frequencies[character] / total_characters

    return probability_dict

def bar_graph(probabilities: dict[str, float]):
    plt.figure(figsize=(13, 7))
    
    probabilities = dict(sorted(probabilities.items(), key=lambda x: x[1], reverse=True))
    
    # Cambiar simbolo de espacio
    display_labels = []
    for char in probabilities.keys():
        if char == ' ':
            display_labels.append('␣')
        else:
            display_labels.append(char)
    
    # Gradiente de colores
    alpha = 0.5
    vals_all = np.array(list(probabilities.values()), dtype=float)
    vals_all = np.maximum(vals_all, 0.0)
    norm_all = (vals_all - vals_all.min())**alpha
    norm_all = norm_all / (norm_all.max() + 1e-8)
    norm_all = 0.2 + 0.8 * norm_all   # Piso 0.2 para que nada quede demasiado claro
    colors_all = plt.cm.Blues(norm_all)
    
    plt.bar(display_labels, probabilities.values(), label='Probabilidad', color=colors_all)
    
    plt.xlabel('Caracteres')
    plt.ylabel('Probabilidad')
    plt.title('Gráfico de Barras')
    plt.yscale('log')

    try:
        from matplotlib.ticker import FuncFormatter

        def decimal_formatter(x, pos):
            # Formatear con hasta 6 decimales, eliminar ceros finales
            s = f"{x:.4f}"
            s = s.rstrip('0').rstrip('.')
            return s if s != '' else '0'

        plt.gca().yaxis.set_major_formatter(FuncFormatter(decimal_formatter))
    except Exception:
        pass

    plt.legend()
    plt.show()

# EJERCICIO 2

# Función provista
def huffman_code(prob_dict: dict[str, float]) -> dict[str, str]:
    """genera un código Huffman a partir de las probabilidades dadas
       en el diccionario "prob_dict"
       Retorna un diccionario con el código de Huffman para cada símbolo"""
    # Crear lista de nodos: cada nodo es (probabilidad, símbolo o subárbol)
    nodes = [(p, {char: ''}) for char, p in prob_dict.items()]

    while len(nodes) > 1:
        # Ordenar nodos por probabilidad
        nodes = sorted(nodes, key=lambda x: x[0])

        # Tomar los dos de menor probabilidad
        p1, c1 = nodes.pop(0)
        p2, c2 = nodes.pop(0)

        # Agregar prefijo 0 al primer subárbol y 1 al segundo
        c1 = {k: '0' + v for k, v in c1.items()}
        c2 = {k: '1' + v for k, v in c2.items()}

        # Fusionar los dos subárboles
        new_node = (p1 + p2, c1 | c2)
        nodes.append(new_node)

    # Retornar el diccionario con el código final (nodes: list[tuple[int, dict]])
    return nodes[0][1]

def text_to_code(file: str, huffman_dict: dict[str, str]) -> str:
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    bits = ''

    for character in content:
        bits += huffman_dict[character]
    
    return bits

def average_length(prob_dict: dict[str, float], huffman_dict: dict[str, str]) -> float:
    length = 0

    for character in prob_dict:
        length += (prob_dict[character] * len(huffman_dict[character]))

    return length

# En una codificación uniforme, todos los símbolos deben representarse con la misma cantidad de bits.
# Para poder codificar N símbolos diferentes, necesitas la menor cantidad de bits k tal que 2^k >= N.
# Ejemplo:
# - Si tienes 2 símbolos, necesitas 1 bit (2¹ = 2).
# - Si tienes 4 símbolos, necesitas 2 bits (2² = 4).
# - Si tienes 5 símbolos, necesitas 3 bits (2³ = 8, porque 2²=4 no alcanza).
# Por eso, la cantidad de bits necesaria es el menor entero mayor o igual a log2(N), es decir, ceil(log2(N)).

def uniform_code_length(prob_dict: dict[str, float]) -> int:
    N = len(prob_dict)
    return math.ceil(math.log2(N))

def memory_reduction(l_huffman: float, bits_uniform: int, total_chars: int) -> float:
    bits_huffman = l_huffman * total_chars
    bits_uniform = bits_uniform * total_chars
    reduction = ((bits_uniform - bits_huffman) / bits_uniform) * 100

    return reduction

# EJERCICIO 3

def entropy(prob_dict: dict[str, float]):
    H = 0
    
    for prob in prob_dict.values():
        H += prob * math.log2(prob)
    
    return -H

# Comparación H(X) vs L para un texto
def compare_entropy_and_length(text_path):
    prob_dict = probability_mass(text_path)
    H = entropy(prob_dict)
    huff_dict = huffman_code(prob_dict)
    L = average_length(prob_dict, huff_dict)

    gap = L - H  # redundancia promedio
    efficiency = (H / L) if L > 0 else float('nan')

    print(f"{'='*60}")
    print(f"Ejercicio 3 — Análisis de: {text_path}")
    print(f"{'='*60}")
    print(f"Símbolos distintos (N): {len(prob_dict)}")
    print(f"Entropía H(X): {H:.6f} bits/símbolo")
    print(f"Longitud promedio L (Huffman): {L:.6f} bits/símbolo")
    print(f"Brecha L - H: {gap:.6f} bits/símbolo")
    print(f"Eficiencia H/L: {efficiency:.4f}")
    print(f"Cota de Huffman (H ≤ L < H+1): {H <= L < H + 1}\n")

# EJERCICIO 4

def normalized_entropy(prob_dict):
    nu = entropy(prob_dict)/math.log2(len(prob_dict))
    return nu

def analyze_normalized_entropy(text_path):
    prob = probability_mass(text_path)
    H = entropy(prob)
    N = len(prob)
    eta = normalized_entropy(prob)
    
    print(f"{'='*60}")
    print(f"Ejercicio 4 — Entropía normalizada: {text_path}")
    print(f"{'='*60}")
    print(f"Símbolos distintos (N): {N}")
    print(f"Entropía H(X): {H:.6f} bits/símbolo")
    print(f"Entropía normalizada η = H/log2(N): {eta:.6f} (adimensional, 0–1)")
    
    if eta > 0.95:
        comment = "Distribución cercana a uniforme (muy dispersa)."
    elif eta < 0.6:
        comment = "Distribución muy concentrada (alta redundancia)."
    else:
        comment = "Distribución intermedia: cierta estructura, cierta aleatoriedad."
        
    print(f"Interpretación: {comment}\n")