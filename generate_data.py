import random
import os
def generate_modular_data(p, operator):
    data = []
    for a in range(p):
        for b in range(p):
            if operator == '+':
                c = (a + b) % p
            elif operator == '-':
                c = (a - b) % p
            elif operator == '/':
                if b == 0:
                    continue  # no inverse for 0
                b_inv = pow(b, p - 2, p)  # Fermat's Little Theorem
                c = (a * b_inv) % p
            else:
                continue
            data.append(f"{a} {operator} {b} = {c}")
    return data

def write_splits(data, prefix, p):
    random.shuffle(data)
    n = len(data)
    train, val, test = data[:int(0.8 * n)], data[int(0.8 * n):int(0.9 * n)], data[int(0.9 * n):]

    os.makedirs("data", exist_ok=True)
    with open(f"data/train_{prefix}_{p}.txt", "w") as f:
        f.write("\n".join(train))
    with open(f"data/val_{prefix}_{p}.txt", "w") as f:
        f.write("\n".join(val))
    with open(f"data/test_{prefix}_{p}.txt", "w") as f:
        f.write("\n".join(test))

def generate_all_data():
    for p in [97, 113]:
        for operator, name in [('+', 'addition'), ('-', 'subtraction'), ('/', 'division')]:
            data = generate_modular_data(p, operator)
            write_splits(data, name, p)
            print(f"Generated {len(data)} samples for {name} mod {p}")

generate_all_data()

