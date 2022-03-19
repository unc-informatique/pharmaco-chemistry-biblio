"""Exemple avec les generateurs"""

from time import sleep, perf_counter
from itertools import takewhile, islice

DELAY = 0.5

def delay(n):
    """delay"""
    sleep(DELAY)
    return n


print("start: comprehension")
start = perf_counter()
l = [delay(n) for n in range(3)]
print(f">>> {perf_counter() - start} sec.")

print("start: print comprehension")
start = perf_counter()
print(l)
print(f">>> {perf_counter() - start} sec.")

print("start: generator")
start = perf_counter()
g = (delay(n) for n in range(3))
print(f">>> {perf_counter() - start} sec.")

print("start: print generator")
start = perf_counter()
for i in g:
    print(i)
print(f">>> {perf_counter() - start} sec.")

def fib():
    """fibonacci numbers generators"""
    a, b = 0, 1
    yield a
    yield b
    while True:
        b, a = a + b, b
        yield b


print("start: fib islice")
for i in islice(fib(), 10):
    print(i)
    sleep(DELAY)

print("start: fib takewhile")
for i in takewhile(lambda x: x < 100, fib()):
    print(i)
    sleep(DELAY)

print("end")
