import math
import matplotlib.pyplot as plt

# EJERCICIO 1

def frequency_character(archive):

    frequencies = {}

    with open(archive, 'r', encoding='utf-8') as arc: # encoding para leer tildes o ñ

        content = arc.read()

        for character in content:
            if character in frequencies:
                frequencies[character] += 1
            else:
                frequencies[character] = 1
    
    return frequencies
    
def probability_mass(text):

    frequencies = frequency_character(text)

    probability_dict = {}

    total_characters = sum(frequencies.values())

    for character in frequencies.keys():

        probability_dict[character] = frequencies[character] / total_characters

    return probability_dict

def bar_graph(probabilities):

    plt.bar(probabilities.keys(), probabilities.values())

    plt.xlabel('Caracteres')
    plt.ylabel('Probabilidad')
    plt.title('Gráfico de Barras')

    plt.show()

# probabilities = probability_mass('texts/tabla.txt')

# bar_graph(probabilities)

# EJERCICIO 2

# Función provista
def huffman_code(prob_dict):
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

def text_to_code(archive, huffman_dict):

    with open(archive, 'r', encoding='utf-8') as arc:

        content = arc.read()
    
    bits = ''

    for character in content:
        bits += huffman_dict[character]
    
    return bits

def average_length(prob_dict, huffman_dict):

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

def uniform_code_length(prob_dict):
    N = len(prob_dict)
    return math.ceil(math.log2(N))

def memory_reduction(l_huffman, bits_uniform, total_chars):

    bits_huffman = l_huffman * total_chars
    bits_uniform = bits_uniform * total_chars
    reduction = ((bits_uniform - bits_huffman) / bits_uniform) * 100

    return reduction

# if __name__ == "__main__":
#     textos = ['texts/adn.txt', 'texts/mitología.txt', 'texts/tabla.txt']
#     for archivo in textos:
#         print(f"\n{'='*60}")
#         print(f"Análisis de: {archivo}")
#         print(f"{'='*60}")
#         prob_dict = probability_mass(archivo)
#         huff_dict = huffman_code(prob_dict)
#         bits = text_to_code(archivo, huff_dict)
#         L = average_length(prob_dict, huff_dict)
#         bits_uniform = uniform_code_length(prob_dict)
#         with open(archivo, 'r', encoding='utf-8') as f:
#             total_chars = len(f.read())
#         reduction = memory_reduction(L, bits_uniform, total_chars)
#         print(f"Longitud promedio de bits (Huffman): {L:.4f}")
#         print(f"Bits por símbolo (uniforme): {bits_uniform}")
#         print(f"Total de caracteres: {total_chars}")
#         print(f"Bits totales (Huffman): {L*total_chars:.2f}")
#         print(f"Bits totales (uniforme): {bits_uniform*total_chars}")
#         print(f"Reducción de memoria: {reduction:.2f}%")
#         print(f"Primeros 100 bits codificados: {bits[:100]}...")

# EJERCICIO 3

def entropy(prob_dict):
    H = 0
    for prob in prob_dict.values():
        H += prob * math.log2(prob)
    return -H

# # Comparación H(X) vs L para un texto
# def compare_entropy_and_length(text_path):
#     prob_dict = probability_mass(text_path)
#     H = entropy(prob_dict)
#     huff_dict = huffman_code(prob_dict)
#     L = average_length(prob_dict, huff_dict)

#     gap = L - H  # redundancia promedio
#     efficiency = (H / L) if L > 0 else float('nan')

#     print(f"\n{'='*60}")
#     print(f"Ejercicio 3 — Análisis de: {text_path}")
#     print(f"{'='*60}")
#     print(f"Símbolos distintos (N): {len(prob_dict)}")
#     print(f"Entropía H(X): {H:.6f} bits/símbolo")
#     print(f"Longitud promedio L (Huffman): {L:.6f} bits/símbolo")
#     print(f"Brecha L - H: {gap:.6f} bits/símbolo")
#     print(f"Eficiencia H/L: {efficiency:.4f}")
#     print(f"Cumple cota de Huffman (H ≤ L < H+1): {H <= L < H + 1}")


# if __name__ == "__main__":
#     textos = ['texts/adn.txt', 'texts/mitología.txt', 'texts/tabla.txt']
#     for archivo in textos:
#         compare_entropy_and_length(archivo)


# EJERCICIO 4

def normalized_entropy(prob_dict):
    nu = entropy(prob_dict)/math.log2(len(prob_dict))
    return nu

# def analyze_normalized_entropy(text_path):
#     prob = probability_mass(text_path)
#     H = entropy(prob)
#     N = len(prob)
#     eta = normalized_entropy(prob)
#     print(f"\n{'='*60}")
#     print(f"Ejercicio 4 — Entropía normalizada: {text_path}")
#     print(f"{'='*60}")
#     print(f"Símbolos distintos (N): {N}")
#     print(f"Entropía H(X): {H:.6f} bits/símbolo")
#     print(f"Entropía normalizada η = H/log2(N): {eta:.6f} (adimensional, 0–1)")
#     if eta > 0.95:
#         comment = "Distribución cercana a uniforme (muy dispersa)."
#     elif eta < 0.6:
#         comment = "Distribución muy concentrada (alta redundancia)."
#     else:
#         comment = "Distribución intermedia: cierta estructura, cierta aleatoriedad."
#     print(f"Interpretación: {comment}")


# if __name__ == "__main__":
#     textos = ['texts/adn.txt', 'texts/mitología.txt', 'texts/tabla.txt']
#     for archivo in textos:
#         analyze_normalized_entropy(archivo)

