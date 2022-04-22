import random
import numpy as np

def random_vector(n):
   A = np.random.randint(0, 100, size=(n))
   A.reverse()
   print (A)
   return A

  #  print(A)
  #  random.shuffle(A)
   # return A

def random_vector_incr(n):
    A = []
    for i in range(n):
        A.append(0 + i)
    print(A)
    return A

def random_vector_rev(n):
    A = []
    for i in range(n):
        x = random.randint(0, n)
        A.append(x)
    A.reverse()
    print(A)
    return A


if __name__ == "__main__":
    random_vector_rev(10)

