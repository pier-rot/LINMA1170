from numba import njit, prange
import numpy as np

## Helper functions
# Vector dot product
@njit(parallel=True, cache=True)
def dot(v1 : np.ndarray, v2 : np.ndarray):
    if v1.shape[0] != v2.shape[0] :
        raise ValueError(f"Invalid shape : {v1.shape[0]} != {v2.shape[0]}")
    
    v1 = v1.astype(np.complex64)
    v2 = v2.astype(np.complex64)
    
    sum = 0
    for i in prange(v1.shape[0]):
        sum += v1[i]*v2[i]
    
    return sum

# Matrix product   
@njit(parallel=True, cache=True)
def mprod(m1 : np.ndarray, m2 : np.ndarray) -> np.ndarray:
    if m1.shape[1] != m2.shape[0]:
        raise ValueError(f"m1 has {m1.shape[1]} columns while m2 has {m2.shape[0]} rows")

    m1 = m1.astype(np.complex64)
    m2 = m2.astype(np.complex64)

    if m1.ndim == 1:
        m1 = np.reshape((1,m1.size))
    if m2.ndim == 1:
        m2 = np.reshape((1,m2.size))

    m,n = m1.shape
    _,p = m2.shape

    m3 = np.zeros((m,p),dtype=np.complex64)

    for i in prange(m):
        for j in prange(p):
            for k in prange(n):
                m3[i,j] += m1[i,k] * m2[k,j]

    return m3

@njit(parallel=True, cache=True)
def vnorm(v : np.ndarray):
    if v.ndim != 1:
        raise ValueError(f"{v} is not a vector")
    
    sum : np.float64 = 0
    n = v.shape[0]

    for i in prange(n):
        sum += v[i]*v[i]

    return np.sqrt(sum)

## Main Functions
# Transformation sous forme Hessenberg
@njit(parallel=False, cache=True)
def hessenberg(A : np.ndarray , P : np.ndarray):
    """Transformation of A in upper Hessenberg form.
    This is done in-place to reduce memory usage.

    Args:
        A (np.ndarray (with np.complex_ entries)): Contains n x n matrix A
        P (np.ndarray (with np.complex_ entries)): Un-initialised array which contains n x n unitary transformation matrix P
    """

# Transformation QR
@njit(parallel=False, cache=True)
def step_QR(H : np.ndarray ,U : np.ndarray ,m : int) -> int:
    """A unitary transformation Q is applied on the left and the right of H which transforms H into RQ for which QR is the QR-factorization of H.
    This is done in-place to reduce memory usage.

    Args:
        H (np.ndarray (with np.complex_ entries)): A complex n x n matrix in Hessenberg form
        U (np.ndarray (with np.complex_ entries)): A complex n x n matrix which contains U such that H = U* A U
        m (int): The dimension of the active matrix
    Returns:
        m_new (int) : New dimension of active matrix after an eigenvalue is found
    """
    m_new = 0
    return m_new

# Transformation QR with shifts
@njit(parallel=False, cache=True)
def step_qr_shift(H : np.ndarray, U : np.ndarray , m : int):
    """Introducing the Wilksinson shift σ, we compute the QR factorization of H - σI instead of H.
    This is done in-place to reduce memory usage.

    Args:
        H (np.ndarray (with np.complex_ entries)): A complex n x n matrix in Hessenberg form
        U (np.ndarray (with np.complex_ entries)): A complex n x n matrix which contains U such that H = U* A U
        m (int): The dimension of the active matrix
    """

# Algorithme QR
@njit(parallel=False, cache=True)
def solve_qr(A : np.ndarray, use_shifts : bool , eps : float , max_iter : int) -> tuple[np.ndarray, int]:
    """Computes the QR factorization such that A = U* T U.
    This is done in-place to reduce memory usage.

    Args:
        A (np.ndarray (with np.complex_ entries)): Contains n x n matrix A as input and n x n matrix T as output
        use_shifts (boolean): Is set to True to use shifts and False otherwise
        eps (float): Is a threshold for which we say that the algorithm has converged if the abs. val. of the subdiagonal entries of A is less than eps
        max_iter (int): Is an upper limit to the number of iterations allowed
    
    Returns:
        U (np.ndarray (with np.complex_ entries)) : Contains the unitary transformation U such that A = U* T U
        k (int) : Number of iterations necessary or -1 if max_iter is exceeded
    """
    U = A
    k = 0

    return U,k