# Caso de prueba 1
msgs_1 = [
    'Mañana vuelo a francia',
    'Feliz inicio de semestre',
    'Tengo un perrito en casa'
]

vocab_1 = [
    'feliz',
    'perrito',
    'casa',
    'comida',
    'semestre',
    'puerta',
    'vuelo'
]
expected_output_1 = [
    [0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 0, 0],
    [0, 1, 1, 0, 0, 0, 0]
]

# Caso de prueba 2
msgs_2 = [
    'Hola, feliz día',
    'No tengo tiempo para la comida',
    'Mi casa tiene una puerta grande'
]

vocab_2 = [
    'feliz',
    'perrito',
    'casa',
    'comida',
    'semestre',
    'puerta',
    'vuelo'
]

expected_output_2 = [
    [1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0]
]

# Caso de prueba 3 (sin apariciones en el vocabulario)
msgs_3 = [
    'Esto es una prueba',
    'Otro mensaje de prueba',
    'Nada relevante aquí'
]

vocab_3 = [
    'feliz',
    'perrito',
    'casa',
    'comida',
    'semestre',
    'puerta',
    'vuelo'
]

expected_output_3 = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0]
]


test_cases = [
    {
        "messages": msgs_1,
        "vocab": vocab_1,
        "expected_output": expected_output_1,
    },
    {
        "messages": msgs_2,
        "vocab": vocab_2,
        "expected_output": expected_output_2,
    },
    {
        "messages": msgs_3,
        "vocab": vocab_3,
        "expected_output": expected_output_3,
    }
]