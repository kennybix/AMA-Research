import itertools
def perm(n, seq):
    sequence = []
    for p in itertools.product(seq, repeat=n):
        #print("".join(p))
        y = ",".join(p)
        sequence.append(y)
    return sequence