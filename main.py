########    INSERTION-SORT VERSUS MERGE-SORT    ########

import math
import random
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import numpy as np

################################################################
#                        INSERTION-SORT                        #


# -efficente nel odinamento di pochi elementi, stesso funzionamento dell'ordinamento delle carte da gioco
# -parto con una mano vuota, prendo una cartta per volta dal tavolo e la inserisconella posizione corretta delle carte che ho in mano via via
# -confrontando la carta da inserire con le singole carte da dx verso sx
# -in ogni momento le carte in mano sono ordinate

#   INPUT -> array A di n elementi da ordinare
# a ogni passo j ho un numero in più nella giusta posizione

# caso ottimo-- array ordinato O(n)
# caso pessimo-- array ordinato al contrario O(n^2) (anche caso medio)

timelimit = 1000  # 16.6 mins


def Insertion_Sort(A):
    start = timer()
    n = len(A)
    for j in range(1, n):
        if timer() - start > timelimit:
            print("ERROR limit time")
        else:
            key = A[j]
            i = j - 1
            while i >= 0 and A[i] > key:
                A[i + 1] = A[i]
                i = i - 1
            A[i + 1] = key
    end = timer()
    time = end - start
    return float(time)


##############################################################
#                       MERGE-SORT                        #

# -alg ricorsivo Divide et Impera

def Merge_Sort(A, p, r):
    if p < r:
        q = math.floor((p + r) / 2)  # floor division
        Merge_Sort(A, p, q)
        Merge_Sort(A, q + 1, r)
        Merge(A, p, q, r)


def Merge(A, p, q, r):
    n1 = q - p + 1
    n2 = r - q
    L = []
    R = []
    # crea array L[1...n1+1] e R[1..n2+1]
    for i in range(0, n1):
        L.append(A[p + i])
    for j in range(0, n2):
        R.append(A[q + j + 1])
    L.append(math.inf)
    R.append(math.inf)
    i = j = 0
    for k in range(p, r + 1):
        if L[i] <= R[j]:
            A[k] = L[i]
            i += 1
        else:
            A[k] = R[j]
            j += 1


def CalculateTimeMergeSort(A, p, r):
    start = timer()
    Merge_Sort(A, p, r)
    end = timer()
    time = end - start
    return float(time)


######################################################
#                     VECTOR                         #

def random_vector(n):
    A = np.random.randint(0, n*n, size=n)
    print("random vector: ", A)
    return A


#  print(A)
#  random.shuffle(A)
# return A

def random_vector_incr(n):
    A = np.random.randint(0, n*n, size=n)
    A.sort()
    print("Random vector incr: ", A)
    return A

def random_vector_rev(n):
    A = []
    for i in range(n):
        x = random.randint(0, n*n)
        A.append(x)
    A.sort()
    A.reverse()
    print("Random vector rev: ", A)
    return A

def TestInsertionSortAlgorithm(typeVector):
    MediaInsertion = []
    MediaMerge = []
    TempoInsertion = []
    TempoMerge = []

    ind = 0
    dim = 10
    d = []

    while ind < 40:
        d.append(dim)
        if typeVector == "Random":
            A = random_vector(dim)
        if typeVector == "RandomIncr":
            A = random_vector_incr(dim)
        if typeVector == "RandomDecr":
            A = random_vector_rev(dim)
        i = 0
        while i < 5:
            B = A
            TempoInsertion.append(Insertion_Sort(B))
            TempoMerge.append(CalculateTimeMergeSort(B, 0, dim-1))
            i += 1
        dimensione = len(TempoMerge)
        sumI = 0
        sumM = 0
        for i in range(0, dimensione):
            sumI += TempoInsertion[i]
            sumM += TempoMerge[i]

        mediaI = sumI/dimensione
        mediaM = sumM/dimensione
        MediaInsertion.append(mediaI)
        MediaMerge.append(mediaM)
        ind += 1
        dim += 100
    plt.plot(d, MediaInsertion, label="InsertionSort")
    plt.plot(d, MediaMerge, label="MergeSort")
    plt.legend()
    plt.ylabel('Secondi')
    plt.xlabel('Ordinamento con vettore' + typeVector)
    plt.show()

#    def TestVectShuffle(typeVector):
#    timeInsSort = []
#    timeMeSort = []
#    sumInsertionTime = 0.0
#    sumMergeTime = 0.0
#    ind = 0
#    dim = 10


#    yIM = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#    while ind < 200:
#        if typeVector == "Random":
#            A = random_vector(dim * 2)
#        if typeVector == "RandomIncr"
#        A = random_vector_incr(dim * 2)
#        if typeVector == "RandomDecr":
#            A = random_vector_rev(dim * 2)
#        i = 0
#        yIM[ind] = ind
#        while( i < 5):
#           B = A

#            timeInsSort.append(Insertion_Sort(B))
#            sumInsertionTime += timeInsSort[i]
#            p = 0
#            timeMeSort.append(CalculateTimeMergeSort(B, p, dim - 1))
#            sumMergeTime += timeMeSort[i]
#        i += 1
#    sumInsertionTime = sumInsertionTime / len(yIM)
#    sumMergeTime = sumMergeTime / len(yIM)
#    xI = timeInsSort
#    xM = timeMeSort
#    print("XI:", xI)
#    print("XM:", xM)
#    print("media insertion : ", sumInsertionTime)
#    print("media merge: ", sumMergeTime)

#    plt.title("Ordinamento " + typeVector + " Insertion Sort and Merge Sort")
#    plt.plot(xI, yIM, marker="o", color='red', label="Insetion Sort")
#    plt.plot(xM, yIM, marker="o", color="green", label="Merge Sort")

#    plt.scatter(sumInsertionTime, 5.5, color='purple', label="mediaI", marker="x")
#    plt.scatter(sumMergeTime, 5.5, color="blue", label="mediaM", marker="x")
#    plt.legend()
#    plt.ylabel("numero prova")
#    plt.xlabel("tempo impiegato")
#    plt.show()

    # plt.title("Ordinamento vettore shaffle Merge Sort")
    # plt.plot(xM, yIM, marker="o", color='green')
    # plt.ylabel("numero prova")
    # plt.xlabel("tempo impiegato")
    # plt.show()


if __name__ == "__main__":
    TestInsertionSortAlgorithm("RandomIncr")
