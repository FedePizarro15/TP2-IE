import math
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

PMF = dict[str, float] #! ¿Esto es una PMF?
Frecuency = dict[str, int]
Huffman = dict[str, str]

# EJERCICIO 1

def frequency_character(file: str) -> Frecuency:
    frequencies = {}

    with open(file, 'r', encoding='utf-8') as f: # encoding para leer tildes o ñ
        content = f.read()

        for character in content:
            if character in frequencies:
                frequencies[character] += 1
            else:
                frequencies[character] = 1
    
    return frequencies
    
def probability_mass(file: str) -> PMF:
    frequencies = frequency_character(file)

    pmf = {}

    total_characters = sum(frequencies.values())

    for char in frequencies.keys():
        pmf[char] = frequencies[char] / total_characters

    return pmf

def decimal_formatter(x, pos):
    s = f"{x:.4f}"
    s = s.rstrip('0').rstrip('.')
    return s if s != '' else '0'

def bar_graph(pmf: PMF):
    # Preparar figura con dos subplots lado a lado: lineal (izquierda) y log (derecha)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)

    pmf = dict(sorted(pmf.items(), key=lambda x: x[1], reverse=True))

    # Cambiar símbolo de espacio para el display
    display_labels = []
    for char in pmf.keys():
        display_labels.append('␣' if char == ' ' else char)

    # Gradiente de colores (mismo para ambos gráficos)
    alpha = 0.5
    vals_all = np.array(list(pmf.values()), dtype=float)
    vals_all = np.maximum(vals_all, 0.0)
    norm_all = (vals_all - vals_all.min())**alpha
    norm_all = norm_all / (norm_all.max() + 1e-8)
    norm_all = 0.2 + 0.8 * norm_all   # Piso 0.2 para que nada quede demasiado claro
    colors_all = plt.cm.Blues(norm_all)

    # Gráfico lineal (eje izquierdo)
    ax = axes[0]
    ax.bar(display_labels, pmf.values(), label='Probabilidad', color=colors_all)
    ax.set_xlabel('Caracteres')
    ax.set_ylabel('Probabilidad')
    ax.set_title('Probabilidad (Lineal)')
    ax.legend()

    # Gráfico logarítmico (eje derecho)
    ax2 = axes[1]
    ax2.bar(display_labels, pmf.values(), label='Probabilidad', color=colors_all)
    ax2.set_xlabel('Caracteres')
    ax2.set_ylabel('Probabilidad')
    ax2.set_title('Probabilidad (Logarítmica)')
    ax2.set_yscale('log')
    ax2.yaxis.set_major_formatter(FuncFormatter(decimal_formatter))
    ax2.legend()

    # Ajustes estéticos
    plt.setp(axes, xticks=range(len(display_labels)), xticklabels=display_labels)
    fig.tight_layout()
    plt.show()

# EJERCICIO 2

# Función provista
def huffman_code(pmf: PMF) -> Huffman:
    """genera un código Huffman a partir de las probabilidades dadas
       en el diccionario "prob_dict"
       Retorna un diccionario con el código de Huffman para cada símbolo"""
    # Crear lista de nodos: cada nodo es (probabilidad, símbolo o subárbol)
    nodes = [(p, {char: ''}) for char, p in pmf.items()]

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

def text_to_code(file: str, huffman_dict: Huffman) -> str:
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    bits = ''

    for character in content:
        bits += huffman_dict[character]
    
    return bits

def average_length(prob_dict: PMF, huffman_dict: Huffman) -> float:
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

def uniform_code_length(pmf: PMF) -> int:
    N = len(pmf)
    return math.ceil(math.log2(N))

def memory_reduction(l_huffman: float, bits_uniform: int, total_chars: int) -> float:
    bits_huffman = l_huffman * total_chars
    bits_uniform = bits_uniform * total_chars
    reduction = ((bits_uniform - bits_huffman) / bits_uniform) * 100

    return reduction

def exercise_2(file: str) -> None:
    print(f"{'='*60}")
    print(f"Análisis de: {file}")
    print(f"{'='*60}")
    
    prob_dict = probability_mass(file)
    huff_dict = huffman_code(prob_dict)
    bits = text_to_code(file, huff_dict)
    L = average_length(prob_dict, huff_dict)
    bits_uniform = uniform_code_length(prob_dict)
    
    with open(file, 'r', encoding='utf-8') as f:
        total_chars = len(f.read())
    
    reduction = memory_reduction(L, bits_uniform, total_chars)
    
    print(f"Longitud promedio de bits (Huffman): {L:.4f}")
    print(f"Bits por símbolo (uniforme): {bits_uniform}")
    print(f"Total de caracteres: {total_chars}")
    print(f"Bits totales (Huffman): {L*total_chars:.0f}")
    print(f"Bits totales (uniforme): {bits_uniform*total_chars}")
    # Mostrar la reducción de forma segura: si bits_uniform es 0, informamos
    # que no aplica (p. ej. sólo un símbolo en el texto).
    if bits_uniform == 0:
        print("Reducción de memoria: N/A (solo un símbolo en el alfabeto)")
    else:
        print(f"Reducción de memoria: {reduction:.2f}%")
    print(f"Primeros 100 bits codificados: {bits[:100]}...\n")
    
# EJERCICIO 3

def entropy(pmf: PMF) -> float:
    H = 0
    
    for prob in pmf.values():
        H += prob * math.log2(prob)
    
    return -H

# Comparación H(X) vs L para un texto
def compare_entropy_and_length(file: str) -> None:
    pmf = probability_mass(file)
    H = entropy(pmf)
    huff_dict = huffman_code(pmf)
    L = average_length(pmf, huff_dict)

    gap = L - H  # redundancia promedio
    efficiency = (H / L) if L > 0 else float('nan')

    print(f"{'='*60}")
    print(f"Ejercicio 3 — Análisis de: {file}")
    print(f"{'='*60}")
    
    print(f"Símbolos distintos (N): {len(pmf)}")
    print(f"Entropía H(X): {H:.6f} bits/símbolo")
    print(f"Longitud promedio L (Huffman): {L:.6f} bits/símbolo")
    print(f"Brecha L - H: {gap:.6f} bits/símbolo")
    print(f"Eficiencia H/L: {efficiency:.4f}")
    print(f"Cota de Huffman (H ≤ L < H+1): {H <= L < H + 1}\n")

# EJERCICIO 4

def normalized_entropy(pmf: PMF) -> float:
    nu = entropy(pmf)/math.log2(len(pmf))
    return nu

def analyze_normalized_entropy(file: str) -> None:
    pmf = probability_mass(file)
    H = entropy(pmf)
    N = len(pmf)
    eta = normalized_entropy(pmf)
    
    print(f"{'='*60}")
    print(f"Ejercicio 4 — Entropía normalizada: {file}")
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