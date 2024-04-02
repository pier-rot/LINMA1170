

# Transformation sous forme Hessenberg
def hessenberg(A,P):
    """Transformation of A in upper Hessenberg form.
    This is done in-place to reduce memory usage.

    Args:
        A (np.ndarray (with np.complex_ entries)): Contains n x n matrix A
        P (np.ndarray (with np.complex_ entries)): Un-initialised array which contains n x n unitary transformation matrix P
    """
    return 0

# Transformation QR
def step_QR(H,U,m):
    """A unitary transformation Q is applied on the left and the right of H which transforms H into RQ for which QR is the QR-factorization of H.
    This is done in-place to reduce memory usage.

    Args:
        H (np.ndarray (with np.complex_ entries)): A complex n x n matrix in Hessenberg form
        U (np.ndarray (with np.complex_ entries)): A complex n x n matrix which contains U such that H = U* A U
        m (int): The dimension of the active matrix
    """
    return 0

# Transformation QR with shifts
def step_qr_shift(H, U, m):
    """Introducing the Wilksinson shift σ, we compute the QR factorization of H - σI instead of H.
    This is done in-place to reduce memory usage.

    Args:
        H (np.ndarray (with np.complex_ entries)): A complex n x n matrix in Hessenberg form
        U (np.ndarray (with np.complex_ entries)): A complex n x n matrix which contains U such that H = U* A U
        m (int): The dimension of the active matrix
    """

# Algorithme QR
def solve_qr(A, use_shifts,eps,max_iter):
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

    return U,k