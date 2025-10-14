import matplotlib.pyplot as plt

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