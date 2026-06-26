import numpy as np

N = 1000000

good = []

for _ in range(N):

    a = np.random.uniform(0.001,50)
    b = np.random.uniform(0.001,50)
    c = np.random.uniform(0.001,50)
    d = np.random.uniform(0.001,50)
    h = np.random.uniform(0.001,50)

    A = c*h
    B = a/d + 1 - c*h
    C = a*h + b - 1

    if A <= 0:
        continue

    if B >= 0:
        continue

    if C <= 0:
        continue

    disc = B**2 - 3*A*C

    if disc <= 0:
        continue

    if 3*A + B <= 0:
        continue

    if 3*A + 2*B + C <= 0:
        continue

    s1 = (-B - np.sqrt(disc))/(3*A)
    s2 = (-B + np.sqrt(disc))/(3*A)

    if not (0 < s1 < 1 and 0 < s2 < 1):
        continue

    def f(s):
        return A*s**3 + B*s**2 + C*s - b

    if f(s1) <= 0:
        continue

    if f(s2) >= 0:
        continue

    roots = np.roots([A,B,C,-b])

    roots = np.real(
        roots[np.abs(np.imag(roots)) < 1e-8]
    )

    roots.sort()

    if len(roots) != 3:
        continue

    if not np.all((roots > 0) & (roots < 1)):
        continue

    stable_pattern = []

    for s in roots:

        theta = d*h*(1-s)/(d*h+s)

        J11 = (
            -a*h - b - s - c*h*s**2
            + d*h*(1-s)*(1+2*c*h*s)/(d*h+s)
        )

        J12 = (
            -a*h
            + d*h*(1-s)*(1+2*c*h*s)/(d*h+s)
        )

        J21 = (
            s**2 + 2*d*h*s - d*h
        )/(d*h+s)

        J22 = (
            -d*h*(1+d*h)
        )/(d*h+s)

        J = np.array([
            [J11,J12],
            [J21,J22]
        ])

        eig = np.linalg.eigvals(J)

        eig_real = np.real(eig)

        if np.all(eig_real < 0):
            stable_pattern.append("S")

        elif np.any(eig_real > 0) and np.any(eig_real < 0):
            stable_pattern.append("U")

        else:
            stable_pattern.append("X")

    if stable_pattern == ["S","U","S"]:

        good.append({
            "a":a,
            "b":b,
            "c":c,
            "d":d,
            "h":h,
            "A": A,
            "B": B,
            "C": C,
            "roots":roots  
        })

print("Bistable examples found:", len(good))

for g in good[:10]:
    print(g)

if len(good) > 0:

    best = min(
        good,
        key=lambda g: g["d"] * g["h"]
    )

    print("\n=================================")
    print(" Smallest d*h product")
    print("=================================")

    print("d*h =", best["d"] * best["h"])

    print("a =", best["a"])
    print("b =", best["b"])
    print("c =", best["c"])
    print("d =", best["d"])
    print("h =", best["h"])

    print("roots =", best["roots"])
