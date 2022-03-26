from math import sqrt
import numpy as np 


def row_e_dist(Xrow, Yrow):
    sum = 0
    for i in range(0,len(Xrow)):
        xpos = Xrow[i]
        ypos = Yrow[i]
        sum += (xpos-ypos) * (xpos-ypos)
    result = sqrt(sum)
    return result

def euclidean_distances(X, Y):
    """Compute pairwise Euclidean distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Euclidean distance between two rows.
    
    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Euclidean distances between rows of X and rows of Y.
    """
    res_matrix = np.empty([len(X),len(Y)])
    for i in range (0,len(X)):
        for j in range (0,len(Y)):
            res_matrix[i,j] = row_e_dist(X[i],Y[j])
    
    return res_matrix


    raise NotImplementedError()


def row_m_dist(Xrow, Yrow):
    sum = 0
    for i in range(0,len(Xrow)):
        xpos = Xrow[i]
        ypos = Yrow[i]
        sum += abs(xpos-ypos)
    return sum

def manhattan_distances(X, Y):
    """Compute pairwise Manhattan distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Manhattan distance between two rows.
    
    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Manhattan distances between rows of X and rows of Y.
    """
    res_matrix = np.empty([len(X),len(Y)])
    for i in range (0,len(X)):
        for j in range (0,len(Y)):
            res_matrix[i,j] = row_m_dist(X[i],Y[j])
    
    return res_matrix

    raise NotImplementedError()

