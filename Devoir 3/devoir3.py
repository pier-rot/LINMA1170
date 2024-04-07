from numba import njit, prange
import numpy as np

## Constants
c_type = np.complex128

## Helper functions
# Outer product
@njit(parallel=False, cache=True)
def outer(v1 : np.ndarray, v2 : np.ndarray):
    return np.outer(v1,v2)


# General dot product
@njit(cache=True)
def dot(A : np.ndarray, B : np.ndarray) -> np.ndarray:
    if (A.ndim == 1) and (B.ndim == 1) and (A.size == B.size):
        return vdot(A,B)
    
    elif (A.ndim == 2 ) and (B.ndim == 2) and (A.shape[1] == B.shape[0]):
        return MM_prod(A,B)
    
    elif (A.ndim == 1) and (B.ndim == 2) and (A.size == B.shape[0]):
        return VM_prod(A,B)[0]
    
    elif (A.ndim == 2) and (B.ndim == 1) and (A.shape[1] == B.size):
        return MV_prod(A,B)[:,0]
    
    else:
        raise ValueError


# Vector dot product
@njit(parallel=True, cache=True)
def vdot(v1 : np.ndarray, v2 : np.ndarray):
    if v1.shape[0] != v2.shape[0] :
        raise ValueError(f"Invalid shape : {v1.shape[0]} != {v2.shape[0]}")
    v1 = v1.astype(c_type)
    v2 = v2.astype(c_type)
    
    sum = 0
    for i in prange(v1.shape[0]):
        sum += v1[i]*v2[i]
    
    return sum

# Matrix * Matrix product   
@njit(parallel=True, cache=True)
def MM_prod(m1 : np.ndarray, m2 : np.ndarray) -> np.ndarray:
    m1 = m1.astype(c_type)
    m2 = m2.astype(c_type)
    # product of m-by-n matrix m1 and n-by-p matrix m2 with m,n,p != 1
    m,n = m1.shape
    _,p = m2.shape
    
    m3 = np.zeros((m,p), dtype=c_type)
    for i in prange(m):
        for j in prange(p):
            m3[i,j] = vdot(m1[i,:], m2[:,j])

    #m1 = n1
    return m3


# Vector * Matrix product
@njit(cache=True)
def VM_prod(m1 : np.ndarray, m2 : np.ndarray) -> np.ndarray:
    m1 = m1.astype(c_type)
    m2 = m2.astype(c_type)

    a = np.zeros((1,m1.size), dtype=c_type)
    a[0] = m1
    return MM_prod(a,m2)


# Matrix * Vector product
@njit(parallel=True, cache=True)
def MV_prod(m1 : np.ndarray, m2 : np.ndarray) -> np.ndarray:
    m1 = m1.astype(c_type)
    m2 = m2.astype(c_type)

    a = np.zeros((m2.size,1), dtype=c_type)
    for i in prange(m2.size):
        a[i] = np.array([m2[i]],dtype=c_type)
    return MM_prod(m1,a)


# Vector norm
@njit(parallel=True, cache=True)
def vnorm(v : np.ndarray):
    if v.ndim != 1:
        raise ValueError(f"{v} is not a vector")
    
    sum : np.float64 = 0
    n = v.shape[0]

    for i in prange(n):
        sum += v[i]*v[i]

    return np.sqrt(sum)

# Householder vector
@njit(cache=True)
def hhvec(x : np.ndarray) -> np.ndarray:
    e1 = np.zeros_like(x, dtype=c_type)
    e1[0] = 1
    vk = np.sign(x[0])*vnorm(x)*e1 + x
    vk = vk/vnorm(vk)

    return vk


# Givens
@njit(cache=True)
def givens(a : c_type, b : c_type):
    if (b== 0):
        c = 1
        s = 0
    else:
        if (np.abs(b) > np.abs(a)):
            r = -a/b
            s = 1/np.sqrt(1+r*r)
            c = r*s
        else:
            r = -b/a
            c = 1/np.sqrt(1+r*r)
            s = r*c

    return c,s



## Main Functions
# Transformation sous forme Hessenberg
@njit(parallel=True, cache=True)
def hessenberg(A : np.ndarray , P : np.ndarray):
    """Transformation of A in upper Hessenberg form.
    This is done in-place to reduce memory usage.

    Args:
        A (np.ndarray (with np.complex_ entries)): Contains n x n matrix A
        P (np.ndarray (with np.complex_ entries)): Un-initialised array which contains n x n unitary transformation matrix P
    """
    #A = A.astype(c_type)
    #P = P.astype(c_type)
    m = A.shape[0]

    I = np.eye(m,m, dtype=c_type)
    for k in range(m-2):
        x = A[k+1:,k]
        vk = hhvec(x)
        Pk = I[k+1:, k+1:] - 2*outer(vk, np.conjugate(vk))
        P[k+1:, k+1:] = dot(P[k+1:, k+1:], Pk)
        A[k+1:,k:] = dot(Pk, A[k+1:, k:])
        A[:,k+1:] = dot(A[:,k+1:], Pk)



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
    n = H.shape[0]
    Gs = np.zeros((n-1, 2,2),dtype=c_type)

    for k in range(n-1):
        c,s = givens(H[k,k], H[k+1,k])
        Gs[k] = np.array([[c , -s],[s,c]],dtype=c_type)
        H[k:k+2, k:] = dot(Gs[k], H[k:k+2, k:])
    
    for k in range(n-1):
        H[:k+2, k:k+2] = dot(H[:k+2, k:k+2], Gs[k])


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

if __name__ == "__main__":
    m = 10
    A = np.random.randn(m,m).astype(c_type)
    Acopy = A.copy()
    hessenberg(A, np.eye(m,m).astype(c_type))
    #fig,axs = plt.subplots(1,2)
    #axs[0].imshow(np.abs(A), cmap='Blues')
    #axs[1].imshow(np.abs(Acopy), cmap="Blues")
#
    #plt.show()