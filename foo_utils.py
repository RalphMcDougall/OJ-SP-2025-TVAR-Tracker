
import numpy as np
import scipy.special
import scipy.stats
try:
    import cupy as cp
    CUPY_IMPORT_SUCCESSFUL = True
except:
    print("Unable to import CuPy, will use numpy instead")
    CUPY_IMPORT_SUCCESSFUL = False


def get_bounding_box(data : np.ndarray, padding_factor : float = 0):
    min_coords = np.min(np.where(data == None, np.inf, data), axis=0)
    max_coords = np.max(np.where(data == None, -np.inf, data), axis=0)

    dims = max_coords - min_coords 
    min_corner = min_coords - padding_factor * dims 
    max_corner = max_coords + padding_factor * dims

    return min_corner, max_corner 

def array_logaddexp(arr):
    return np.logaddexp.reduce(arr)


def log_effective_num_particles(log_weights):
    return -1 * array_logaddexp(2 * log_weights)


def inv(A):
    # Performing all matrix inversions on the GPU really speeds things up!
    if CUPY_IMPORT_SUCCESSFUL:
        A_gpu = cp.asarray(A)
        A_inv_gpu = np.linalg.inv(A_gpu)
        A_inv = cp.asnumpy(A_inv_gpu)
    else:
        A_inv = np.linalg.inv(A)
    return A_inv


def transpose_last2(A):
    new_order = list(range(len(A.shape)))
    new_order[-2:] = new_order[-2:][::-1]
    return np.transpose(A, new_order)


def evaluate_log_mv_gaussian(x, mean, cov=None, prec=None):
    assert (cov is None or prec is None), "One of the provided covariance and precision must be None."

    diff = x - mean 

    if prec is None:
        inv_cov = inv(cov)
        log_det = np.log(np.linalg.det(2 * np.pi * cov)[:,None])
    else:
        inv_cov = prec 
        log_det = -np.log(np.linalg.det(2 * np.pi * prec)[:,None])

    ans = -0.5 * (transpose_last2(diff) @ inv_cov @ (diff))[:,:,0] - 0.5 * log_det
    return ans 


def sample_mv_gaussian_array(means, covs):
    chol = np.linalg.cholesky(covs)

    flattened_length = means.shape[0] * means.shape[1]
    standard_terms = np.random.normal(loc=0, scale=1, size=(flattened_length))
    standard_terms = standard_terms.reshape(means.shape)

    result = means + chol @ standard_terms

    return result 


def gaussian_product(m1, P1, m2, P2):
    P1_inv = inv(P1)
    P2_inv = inv(P2)
    P_res = inv(P1_inv + P2_inv)
    m_res = P_res @ (P1_inv @ m1 + P2_inv @ m2)

    m_offset = m1 - m2 
    P_const = P1 + P2

    return m_res, P_res, m_offset, P_const


def inv_gamma_update(m_offset, P, alpha, beta):
    new_alpha = alpha + 0.5 * m_offset.shape[1]
    new_beta = beta + 0.5 * (transpose_last2(m_offset) @ inv(P) @ m_offset)[:,:,0] 

    assert new_alpha.shape == alpha.shape, "Inverse gamma update changes the shape of alpha"
    assert new_beta.shape == beta.shape, "Inverse gamma update changes the shape of beta"

    return new_alpha, new_beta 


def get_st_params(m, P, alpha, beta):
    df = 2 * alpha 
    mu = m 
    shape = P * beta[:,:,None] / alpha[:,:,None]

    return df, mu, shape 



class CategoricalDistribution:

    def __init__(self, probs):
        self.probs = probs
        
        self.num_vars = self.probs.shape[0]
        self.num_cats = self.probs.shape[1] 

    def sample(self):
        cumulative_probs = np.cumsum(self.probs, axis=1)

        rvs = np.random.uniform(size=(self.num_vars, 1))

        positions = np.argmax(rvs <= cumulative_probs, axis=1)
        return positions


class MultivariateStudentT:
    # See resource: https://journal.r-project.org/archive/2013/RJ-2013-033/RJ-2013-033.pdf

    def __init__(self, mu, V, df):
        self.mu = mu
        self.V = V 
        self.df = df 

        self.num_vars = mu.shape[0]
        self.dim = mu.shape[1]


    def log_pdf(self, x):
        d = self.dim 

        diff = x - self.mu 
        log_p = -0.5 * (self.df[:,:,None] + d) * np.log(1 + transpose_last2(diff) @ inv(self.V) @ diff / self.df[:,:,None])
        log_Z = scipy.special.loggamma(0.5 * (self.df[:,:,None] + d)) - scipy.special.loggamma(0.5 * self.df[:,:,None]) - 0.5 * d * np.log(np.pi * self.df[:,:,None]) - 0.5 * np.linalg.det(self.V)[:,None,None]
        return log_p + log_Z 
    

    def sample(self):
        chol_facts = np.linalg.cholesky(self.V)
        normal_rvs = np.random.normal(loc=0, scale=1, size=(self.num_vars, self.dim, 1))

        correction_rvs = np.random.chisquare(df=self.df, size=(self.num_vars, 1))
        correction_factor = np.sqrt(self.df / correction_rvs)[:,:,None]
        st_rvs = self.mu + correction_factor * (chol_facts @ normal_rvs)
        return st_rvs
    

    def mse(self, target : np.ndarray):
        assert np.any(self.df > 2), "The variance is only defined for t distributions with more than 2 DoF"
        cov_mat = self.V * self.df[:,:,None] / (self.df[:,:,None] - 2)

        diff = (self.mu - target[None,:,None])
        rmse_vec = np.trace(cov_mat, axis1=1, axis2=2) + (transpose_last2(diff) @ diff).flatten() 
        return rmse_vec.flatten()


def tile_field(field : np.ndarray | None, num_copies : int, copies_adjacent : bool):
    if field is None:
        return None
    
    to_flatten = False 
    if len(field.shape) == 1:
        to_flatten = True 
        field = field[:,None]

    num_particles = field.shape[0]

    original_shape = field.shape
    
    tiled_shape = [1] * len(original_shape)
    tiled_shape[0] = num_copies

    reshaped_shape = list(original_shape)
    reshaped_shape[0] *= num_copies 
    reshaped_shape = tuple(reshaped_shape)

    tiled_field = np.zeros(reshaped_shape)

    if copies_adjacent:
        counts = np.ones(len(field.shape) + 1).astype(int)
        counts[1] = num_copies
        counts = tuple(counts)
        
        tiled_field = np.tile(field[None].swapaxes(0, 1), counts).reshape(reshaped_shape)
    else:
        tiled_field = np.tile(field, tiled_shape)

    if copies_adjacent:
        assert num_copies == 1 or np.allclose(field[0], tiled_field[1]), "Tiled field value does not match corresponding original field value"
        assert num_copies == 1 or np.allclose(tiled_field[::num_copies], tiled_field[1::num_copies]), "Adjacent tiling has not been performed correctly"
    else:
        assert num_copies == 1 or np.allclose(field[0], tiled_field[num_particles]), "Tiled field value does not match corresponding original field value"
        assert num_copies == 1 or np.allclose(tiled_field[:num_particles], tiled_field[num_particles:(2 * num_particles)]), "Non adjacent tiling has not been performed correctly"
    
    if to_flatten:
        tiled_field = tiled_field.flatten()
    
    return tiled_field


def get_rankings(scores : np.ndarray, higher_better):
    res = np.zeros(scores.shape)
    for test_ind in range(scores.shape[1]):
        ordering = np.argsort(scores[:,test_ind])
        for i in range(scores.shape[0]):
            res[ ordering[i], test_ind ] = i
    if higher_better:
        res = scores.shape[0] - res - 1
    return res + 1 
