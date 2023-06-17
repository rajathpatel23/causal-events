import random

def delete_random_tokens(string_input):
    tokens = string_input.split()
    num_remove = 1
    for _ in range(num_remove):
        tokens.pop(random.randint(0, len(tokens)-1))
    return tokens