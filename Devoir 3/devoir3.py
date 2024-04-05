import numba
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter_ns, process_time_ns

## Helper functions
#


## Main Functions
# Transformation sous forme Hessenberg
@numba.jit(nopython=True, parallel=False, cache=True)
def hessenberg(A : np.ndarray , P : np.ndarray):
    """Transformation of A in upper Hessenberg form.
    This is done in-place to reduce memory usage.

    Args:
        A (np.ndarray (with np.complex_ entries)): Contains n x n matrix A
        P (np.ndarray (with np.complex_ entries)): Un-initialised array which contains n x n unitary transformation matrix P
    """

# Transformation QR
@numba.jit(nopython=True, parallel=False, cache=True)
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
@numba.jit(nopython=True, parallel=False, cache=True)
def step_qr_shift(H : np.ndarray, U : np.ndarray , m : int):
    """Introducing the Wilksinson shift σ, we compute the QR factorization of H - σI instead of H.
    This is done in-place to reduce memory usage.

    Args:
        H (np.ndarray (with np.complex_ entries)): A complex n x n matrix in Hessenberg form
        U (np.ndarray (with np.complex_ entries)): A complex n x n matrix which contains U such that H = U* A U
        m (int): The dimension of the active matrix
    """

# Algorithme QR
@numba.jit(nopython=True, parallel=False, cache=True)
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