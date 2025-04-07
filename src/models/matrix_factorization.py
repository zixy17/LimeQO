import numpy as np

def als(X, mask, rank, niters, lambda_):
    """
    Alternating Least Squares algorithm for matrix factorization
    
    Args:
        X: matrix to factorize
        mask: binary mask of observed entries
        rank: rank of the factorization
        niters: number of iterations
        lambda_: regularization parameter
        
    Returns:
        Completed matrix
    """
    n, m = X.shape
    A = np.random.rand(n, rank)
    B = np.random.rand(m, rank)
    
    for _ in range(niters):
        # Update A
        target = X + (1 - mask) * (np.dot(A, B.T))
        A = np.linalg.solve(np.dot(B.T, B) + lambda_ * np.eye(rank), 
                          np.dot(target, B).T).T
        A[A < 0] = 0
        
        # Update B
        target = X + (1 - mask) * (np.dot(A, B.T))
        B = np.linalg.solve(np.dot(A.T, A) + lambda_ * np.eye(rank),
                          np.dot(target.T, A).T).T
        B[B < 0] = 0
    
    return X + (1 - mask) * (np.dot(A, B.T))

def censored_als(X, mask, cutoffs, rank, niters, lambda_):
    """
    Censored Alternating Least Squares algorithm for matrix factorization
    
    Args:
        X: matrix to factorize
        mask: binary mask of observed entries 
        cutoffs: censoring thresholds
        rank: rank of the factorization
        niters: number of iterations
        lambda_: regularization parameter
        
    Returns:
        Completed matrix with censoring applied
    """
    n, m = X.shape
    A = np.random.rand(n, rank)
    B = np.random.rand(m, rank)
    
    for _ in range(niters):
        # Update A with censoring
        target = X + (1 - mask) * (np.dot(A, B.T))
        violations = (target < cutoffs) & (cutoffs > 0)
        target[violations] = cutoffs[violations]
        A = np.linalg.solve(np.dot(B.T, B) + lambda_ * np.eye(rank),
                          np.dot(target, B).T).T
        A[A < 0] = 0
        
        # Update B with censoring
        target = X + (1 - mask) * (np.dot(A, B.T))
        violations = (target < cutoffs) & (cutoffs > 0)
        target[violations] = cutoffs[violations]
        B = np.linalg.solve(np.dot(A.T, A) + lambda_ * np.eye(rank),
                          np.dot(target.T, A).T).T
        B[B < 0] = 0
    
    return X + (1 - mask) * (np.dot(A, B.T)) 