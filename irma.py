from sklvq import GMLVQ
import numpy as np
from imblearn.pipeline import make_pipeline
from utils import cross_validation

def get_GMLVQ_model(solver_type = 'wgd', max_runs = 30, step_size = [1, 2], relevance_correction = None):
    """
    Returns an initialized GMLVQ model, with or without subspace correction

    Args:
        solver_type (str): solver for GMLVQ model
        max_runs (int): Number of epochs of training
        step_size (array): step size for updating the prototypes and relevance matrix, respectively
        relevance_correction [np.ndarray]: The correction matrix

    Returns:
        model (sklvq.GMLVQ): Initialized GMLVQ model 
    """
    model = GMLVQ(
        distance_type="adaptive-squared-euclidean",
        activation_type="identity",
        solver_type=solver_type,
        solver_params={
            "max_runs": max_runs,
            "step_size": step_size,
        },
        random_state=0,
        relevance_correction=relevance_correction,
    )

    return model

def compute_correction_matrix(eigenvectors: np.ndarray, nleading: int = None):
    """
    Computes the correction matrix based on directions (eigenvectors) to be disregarded in feature space

    Args:
        eigenvectors (np.ndarray): Row matrix (eigenvectors on the rows)

    Returns:
        [np.ndarray]: The correction matrix
    """
    if nleading is not None:
        eigenvectors = eigenvectors[0:nleading, :]

    # Need to be 2d else dot doesn't work correctly
    eigenvectors = np.atleast_2d(eigenvectors)

    correction_matrix = np.identity(eigenvectors.shape[1]) - (
        eigenvectors.T.dot(eigenvectors)
    )
    # Check if it is a symmetric matrix
    assert np.allclose(correction_matrix, correction_matrix.T)

    return correction_matrix

def eigendecomposition(decomposable_matrix: np.ndarray):
    """
    
    Parameters
    ----------
    decomposable_matrix : np.ndarray
        2D numpy array representing the trained relevance matrix

    Returns
    -------
    eigenvalues : array
        eigenvalues of the input matrix
    eigenvectors : np.ndarray
        eigenvectors of the input matrix
    omega_hat : np.ndarray
        eigenvectors of the input matrix scaled by the square root of each eigenvalue
    """
    # Needs to be real and symmetric
    assert np.all(np.isreal(decomposable_matrix)) & np.allclose(
        decomposable_matrix, decomposable_matrix.T
    )

    # Eigenvalues and column eigenvectors return in ascending order
    eigenvalues, eigenvectors = np.linalg.eigh(decomposable_matrix)

    # Dealing with numerical errors. (eigenvalues of lambda should not be negative in theory)
    eigenvalues[eigenvalues < 0] = 0

    # Flip (reverse the order to descending) before assigning.
    eigenvalues = np.flip(eigenvalues)

    # eigenvectors are column matrix in ascending order. Flip the columns and transpose the matrix
    # to get the descending ordered row matrix.
    eigenvectors = np.flip(eigenvectors, axis=1).T

    # In literature omega_hat contains the "scaled" eigenvectors.
    omega_hat = np.sqrt(eigenvalues[:, None]) * eigenvectors

    return eigenvalues, eigenvectors, omega_hat

def apply_IRMA(X, y, n_iter = 10, n_eigenvectors_removed = 1, solver_type = 'wgd', step_size=[1, 2]):
    """
    Applies IRMA on the data in X
    
    Parameters
    ----------
    X : ndarray
        2D numpy array containing feature vectors in row-first order
    y : array
        numpy array containing labels for the feature vectors
    n_iter : int
        number of iterations of IRMA to apply
    n_eigenvectors_removed : int
        number of eigenvectors to remove per iteration
    solver_type : str
        solver type for GMLVQ model
    step_size : array
        initial step size for updating the prototypes and relevance matrix

    Returns
    -------
    iterated_eigenvectors : np.ndarray
        the accumulated IRMA eigenvectors in row-first order
    iterated_eigenvalues : np.array
        eigenvalue corresponding to each accumulated eigenvector
    """

    iterated_eigenvectors = np.zeros((n_eigenvectors_removed*n_iter, X.shape[1]))
    iterated_eigenvalues = np.zeros(n_eigenvectors_removed*n_iter)
    iterated_correction_matrix = np.identity(X.shape[1])

    for i in range(n_iter):

        model = get_GMLVQ_model(solver_type = solver_type, step_size = step_size,
                                relevance_correction = iterated_correction_matrix)

        model.fit(X,y)
                           
        eigenvalues, eigenvectors, omega_hat = eigendecomposition(model.lambda_)

        iterated_eigenvectors[i:i+n_eigenvectors_removed, :] = eigenvectors[0:n_eigenvectors_removed]
        iterated_eigenvalues[i:i+n_eigenvectors_removed] = eigenvalues[0:n_eigenvectors_removed]

        iterated_correction_matrix = compute_correction_matrix(iterated_eigenvectors, nleading = n_eigenvectors_removed*(i+1))
     
    return iterated_eigenvectors, iterated_eigenvalues



def cross_validate(X, y, repeated_crossvalidation, iterated_eigenvectors, n_iter=10):
    """

    Parameters
    ----------
    X : ndarray
        2D numpy array containing feature vectors in row-first order
    y : array
        numpy array containing labels for the feature vectors
    repeated_crossvalidation : sklearn.model_selection.RepeatedStratifiedKFold
        sklearn cross validator
    iterated_eigenvectors : int
        number of eigenvectors to remove per iteration
    n_iter : int
        number of iterations of IRMA to apply

    Returns
    -------
    irma_bac : np.array
        average cross-validated balanced accuracy per iteration
    irma_auc : np.array
        average cross-validated auc per iteration
    """

    irma_bac = np.zeros(n_iter)
    irma_auc = np.zeros(n_iter)

    for i in range(n_iter):

        iterated_correction_matrix = compute_correction_matrix(iterated_eigenvectors, nleading = i)
        
        model = get_GMLVQ_model(solver_type = 'wgd', max_runs = 30, step_size = [1, 2],
                                relevance_correction = iterated_correction_matrix)
        
        pipeline = make_pipeline(model)

        (
            cv_lambda,
            cv_auc,
            cv_confmat,
            cv_bac,
            cv_accuracy,
            cv_auc_train,
            cv_bac_train
        ) = cross_validation(pipeline, repeated_crossvalidation, X, y)

        eigenvalues, eigenvectors, omega_hat = eigendecomposition(np.mean(cv_lambda, axis=2))

        irma_bac[i] = np.mean(cv_bac)
        irma_auc[i] = np.mean(cv_auc)

    return irma_bac, irma_auc