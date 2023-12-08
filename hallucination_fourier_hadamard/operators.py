import math

from abc import ABC, abstractmethod

import numpy as np
#import pytorch_radon
import skimage.transform
import torch
import torch.fft
import scipy.linalg
#import torch_cg

#from fastmri_utils.data import transforms
#from fastmri_utils.data.transforms import fftshift, ifftshift


# ----- Utilities -----

def convert_to_bit_str(x, logN):
    'Converts an integer x to its binary representation using logN bits'
    return '{:0{width}b}'.format(x, width=logN)

def reverse_bit_order(x, logN):
    '''
    Converts an integer x to its binary representation using logN bits, reverse the order of the bits and convert the number back to decimal representation.
    
    This function is used to change the ordering of an Ordinary Hadamard matrix
    to a Paley ordered Hadamard matrix.
    '''
    b = convert_to_bit_str(x, logN)
    x_rev  = int(b[::-1], 2) # Reverse the order of 'b' and convert to int.
    return x_rev

def hadamard_paley(N):
    """
    Computes the N × N Hadamard matrix with Paley ordering.  
    N must be a power of 2, i.e., N = 2**p for an integer p.
    """
    log2N = int(round(np.log2(N)))
    H = scipy.linalg.hadamard(N) # Gives the Hadamard matrix with the ordinary ordering   
    idx = np.arange(N)
    for i in range(N):
        idx[i] = reverse_bit_order(idx[i], log2N);
    H = H[idx,:]
    return H


class RadialMaskFunc(object):
    """ Generates a golden angle radial spokes mask.

    Useful for subsampling a Fast-Fourier-Transform.
    Contains radial lines (spokes) through the center of the mask, with
    angles spaced according to the golden angle (~111.25°). The first line
    has angle 0° (horizontal). An offset parameter can be given to skip
    the first `offset*num_lines` lines.

    Parameters
    ----------
    shape : array_like
        A tuple specifying the size of the mask.
    num_lines : int
        Number of radial lines (spokes) in the mask.
    offset : int, optional
        Offset factor for the range of angles in the mask.
    """

    def __init__(self, shape, num_lines, offset=0):
        self.shape = shape
        self.num_lines = num_lines
        self.offset = offset
        self.mask = self._generate_radial_mask(shape, num_lines, offset)

    def __call__(self, shape, seed=None):
        assert (self.mask.shape[0] == shape[-3]) and (
            self.mask.shape[1] == shape[-2]
        )
        return torch.reshape(
            self.mask, (len(shape) - 3) * (1,) + self.shape + (1,)
        )

    def _generate_radial_mask(self, shape, num_lines, offset=0):
        # generate line template and empty mask
        x, y = shape
        d = math.ceil(np.sqrt(2) * max(x, y))
        line = np.zeros((d, d))
        line[d // 2, :] = 1.0
        out = np.zeros((d, d))
        # compute golden angle sequence
        golden = (np.sqrt(5) - 1) / 2
        angles = (
            180.0
            * golden
            * np.arange(offset * num_lines, (offset + 1) * num_lines)
        )
        # draw lines
        for angle in angles:
            out += skimage.transform.rotate(line, angle, order=0)
        # crop mask to correct size
        out = out[
            d // 2 - math.floor(x / 2) : d // 2 + math.ceil(x / 2),
            d // 2 - math.floor(y / 2) : d // 2 + math.ceil(y / 2),
        ]
        # return binary mask
        return torch.tensor(out > 0)


def l2_error(X, X_ref, relative=False, squared=False, use_magnitude=True):
    """ Compute average l2-error of an image over last three dimensions.

    Parameters
    ----------
    X : torch.Tensor
        The input tensor of shape [..., 1, W, H] for real images or
        [..., 2, W, H] for complex images.
    X_ref : torch.Tensor
        The reference tensor of same shape.
    relative : bool, optional
        Use relative error. (Default False)
    squared : bool, optional
        Use squared error. (Default False)
    use_magnitude : bool, optional
        Use complex magnitudes. (Default True)

    Returns
    -------
    err_av :
        The average error.
    err :
        Tensor with individual errors.

    """
    assert X_ref.ndim >= 3  # do not forget the channel dimension

    if X_ref.shape[-3] == 2 and use_magnitude:  # compare complex magnitudes
        X_flat = torch.flatten(torch.sqrt(X.pow(2).sum(-3)), -2, -1)
        X_ref_flat = torch.flatten(torch.sqrt(X_ref.pow(2).sum(-3)), -2, -1)
    else:
        X_flat = torch.flatten(X, -3, -1)
        X_ref_flat = torch.flatten(X_ref, -3, -1)

    if squared:
        err = (X_flat - X_ref_flat).norm(p=2, dim=-1) ** 2
    else:
        err = (X_flat - X_ref_flat).norm(p=2, dim=-1)

    if relative:
        if squared:
            err = err / (X_ref_flat.norm(p=2, dim=-1) ** 2)
        else:
            err = err / X_ref_flat.norm(p=2, dim=-1)

    if X_ref.ndim > 3:
        err_av = err.sum() / np.prod(X_ref.shape[:-3])
    else:
        err_av = err
    return err_av.squeeze(), err

def to_complex(x):
    """ Converts real images to complex by adding a channel dimension. """
    assert x.ndim >= 3 and (x.shape[-3] == 1 or x.shape[-3] == 2)
    # real tensor of shape (1, n1, n2) or batch of shape (*, 1, n1, n2)
    if x.shape[-3] == 1:
        imag = torch.zeros_like(x)
        out = torch.cat([x, imag], dim=-3)
    else:
        out = x
    return out


def rotate_real(x):
    """ Rotates the magnitude of a complex signal into the real channel. """
    assert x.ndim >= 3 and (x.shape[-3] == 2)
    x_rv = torch.zeros_like(x)
    x_rv[..., 0, :, :] = torch.sqrt(x.pow(2).sum(-3))
    return x_rv


def im2vec(x, dims=(-2, -1)):
    """ Flattens last two dimensions of an image tensor to a vector. """
    return torch.flatten(x, *dims)


def vec2im(x, n):
    """ Unflattens the last dimension of a vector to two image dimensions. """
    return x.view(*x.shape[:-1], *n)


def prep_fft_channel(x):
    """ Rotates complex image dimension from channel to last position. """
    x_real = torch.unsqueeze(x[...,0,:,:],-1) 
    if x.shape[-3] == 1:
        x_imag = torch.unsqueeze(torch.zeros_like(x[...,0,:,:]),-1)
    else:
        x_imag = torch.unsqueeze(x[...,1,:,:],-1)
    x = torch.cat([x_real, x_imag], dim=-1)
    return torch.view_as_complex(x)

def unprep_fft_channel(x):
    """ Rotates complex image dimension from last to channel position. """
    if torch.is_complex(x):
        x = torch.view_as_real(x)
    x_real = torch.unsqueeze(x[...,0],-3) 
    x_imag = torch.unsqueeze(x[...,1],-3)
    x = torch.cat([x_real, x_imag], dim=-3)
    return x 

def circshift(x, dim=-1, num=1):
    """ Circular shift by n along a dimension. """
    perm = list(range(num, x.shape[dim])) + list(range(0, num))
    if not dim == -1:
        return x.transpose(dim, -1)[..., perm].transpose(dim, -1)
    else:
        return x[..., perm]


# ----- Thresholding, Projections, and Proximal Operators -----


def _shrink_single(x, thresh):
    """ Soft/Shrinkage thresholding for tensors. """
    return torch.nn.Softshrink(thresh)(x)


def _shrink_recursive(c, thresh):
    """ Soft/Shrinkage thresholding for nested tuples/lists of tensors. """
    if isinstance(c, (list, tuple)):
        return [_shrink_recursive(el, thresh) for el in c]
    else:
        return _shrink_single(c, thresh)


shrink = _shrink_single  # alias for convenience


def proj_l2_ball(x, centre, radius):
    """ Euclidean projection onto a closed l2-ball.

    Parameters
    ----------
    x : torch.Tensor
        The input tensor to project.
    centre : torch.Tensor
        The centre of the ball.
    radius : float
        The radius of the ball. Must be non-negative.

    Returns
    -------
    torch.Tensor
        The projection of x onto the closed ball.
    """
    norm = torch.sqrt((x - centre).pow(2).sum(dim=(-2, -1), keepdim=True))
    radius, norm = torch.broadcast_tensors(radius, norm)
    fac = torch.ones_like(norm)
    fac[norm > radius] = radius[norm > radius] / norm[norm > radius]
    return fac * x + (1 - fac) * centre


# ----- Linear Operator Utilities -----


class LinearOperator(ABC):
    """ Abstract base class for linear (measurement) operators.

    Can be used for real operators

        A : R^(n1 x n2) ->  R^m

    or complex operators

        A : C^(n1 x n2) -> C^m.

    Can be applied to tensors of shape (n1, n2) or (1, n1, n2) or batches
    thereof of shape (*, n1, n2) or (*, 1, n1, n2) in the real case, or
    analogously shapes (2, n1, n2) or (*, 2, n1, n2) in the complex case.

    Attributes
    ----------
    m : int
        Dimension of the co-domain of the operator.
    n : tuple of int
        Dimensions of the domain of the operator.

    """

    def __init__(self, m, n):
        self.m = m
        self.n = n

    @abstractmethod
    def dot(self, x):
        """ Application of the operator to a vector.

        Computes Ax for a given vector x from the domain.

        Parameters
        ----------
        x : torch.Tensor
            Must be of shape to (*, n1, n2) or (*, 2, n1, n2).
        Returns
        -------
        torch.Tensor
            Will be of shape (*, m) or (*, 2, m).
        """
        pass

    @abstractmethod
    def adj(self, y):
        """ Application of the adjoint operator to a vector.

        Computes (A^*)y for a given vector y from the co-domain.

        Parameters
        ----------
        y : torch.Tensor
            Must be of shape (*, m) or (*, 2, m).

        Returns
        -------
        torch.Tensor
            Will be of shape (*, n1, n2) or (*, 2, n1, n2).
        """
        pass

    @abstractmethod
    def inv(self, y):
        """ Application of some inversion of the operator to a vector.

        Computes (A^dagger)y for a given vector y from the co-domain.
        A^dagger can for example be the pseudo-inverse.

        Parameters
        ----------
        y : torch.Tensor
            Must be of shape (*, m) or (*, 2, m).

        Returns
        -------
        torch.Tensor
            Will be of shape (*, n1, n2) or (*, 2, n1, n2).
        """
        pass

    def __call__(self, x):  # alias to make operator callable by using dot
        return self.dot(x)


############################################################
###                 Fourier operators                    ###
############################################################

class Fourier(LinearOperator):
    """ 2D discrete Fourier transform.

    Implements the complex operator C^(n1, n2) -> C^m
    appling the (subsampled) Fourier transform.
    The adjoint is the conjugate transpose. The inverse is the same as adjoint.


    Parameters
    ----------
    mask : torch.Tensor
        The subsampling mask for the Fourier transform.

    """

    def __init__(self, mask):
        m = mask.nonzero().shape[0]
        n = mask.shape[-2:]
        super().__init__(m, n)
        self.mask = mask[0, 0, :, :].bool()

    def dot(self, x):
        """ Subsampled Fourier transform. """
        x = prep_fft_channel(x)
        x = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2,-1), norm='ortho'), dim=(-2,-1))
        full_fft = unprep_fft_channel(x)
        return im2vec(full_fft)[..., im2vec(self.mask)]

    def adj(self, y):
        """ Adjoint is the zeor-filled inverse Fourier transform. """
        masked_fft = torch.zeros(
            *y.shape[:-1], self.n[0] * self.n[1], device=y.device
        )
        masked_fft[..., im2vec(self.mask)] = y
        x = prep_fft_channel(vec2im(masked_fft, self.n))       
        x = torch.fft.ifft2(
            torch.fft.ifftshift(x, dim=(-2,-1)), 
            dim=(-2,-1), 
            norm='ortho'
        )
        return unprep_fft_channel(x)

    def inv(self, y):
        """ Pseudo-inverse a.k.a. zero-filled IFFT. """
        return self.adj(y)

    def tikh(self, rhs, kernel, rho):
        """ Tikhonov regularized inversion.

        Solves the normal equation

            (F*F + rho W*W) x = F*y

        or more generally

            (F*F + rho W*W) x = z

        for a Tikhonov regularized least squares fit, assuming that the
        regularization W*W can be diagonalied by FFTs, i.e.

            W*W = F*D*F

        for some diagonal matrix D.

        Parameters
        ----------
        rhs : torch.Tensor
            The right hand side tensor z, often F*y for some y.
        kernel : torch.Tensor
            The Fourier kernel of W, containing the diagonal elements D.
        rho : float
            The regularization parameter.

        """
        assert rhs.ndim >= 3 and rhs.shape[-3] == 2  # assert complex images
        rhs = prep_fft_channel(rhs)
        fft_rhs = torch.fft.fftshift(
            torch.fft.fft2(rhs, dim=(-2,-1), norm='ortho'),
            dim=(-2,-1)
        )
        combined_kernel = prep_fft_channel(
            to_complex(self.mask.unsqueeze(0).float().to(rhs.device))
        ) + rho * kernel.to(rhs.device)
        fft_div = fft_rhs/combined_kernel

        x = torch.fft.ifft2(
            torch.fft.ifftshift(fft_div, dim=(-2,-1)), 
            dim=(-2,-1), 
            norm='ortho'
        )

        return unprep_fft_channel(x)

class LearnableFourier1D(torch.nn.Module):
    """ Learnable 1D discrete Fourier transform.

    Implements a complex operator C^n -> C^n, which is learnable but
    initialized as the Fourier transform.


    Parameters
    ----------
    n : int
        Dimension of the domain and range of the operator.
    dim : int, optional
        Apply the 1D operator along specified axis for inputs with multiple
        axis. (Default is last axis)
    inverse : bool, optional
        Use the discrete inverse Fourier transform as initialization instead.
        (Default False)
    learnable : bool, optional
        Make operator learnable. Otherwise it will be kept fixed as the
        initialization. (Default True)

    """

    def __init__(self, n, dim=-1, inverse=False, learnable=True):
        super(LearnableFourier1D, self).__init__()
        
        self.n = n
        self.dim = dim
        eye_n = torch.stack([torch.eye(n), torch.zeros(n, n)], dim=-1)
        complex_eye_n = torch.view_as_complex(eye_n)
        
        if inverse:
            fft_n = torch.fft.ifft(torch.fft.ifftshift(complex_eye_n, dim=-1), dim=-2, norm='ortho')
        else:
            fft_n = torch.fft.fftshift(torch.fft.fft(complex_eye_n, dim=-1, norm='ortho'), dim=-2)
        
        fft_n = torch.view_as_real(fft_n)
        fft_real_n = fft_n[..., 0]
        fft_imag_n = fft_n[..., 1]
        fft_matrix = torch.cat(
            [
                torch.cat([fft_real_n, -fft_imag_n], dim=1),
                torch.cat([fft_imag_n, fft_real_n],  dim=1),
            ],
            dim=0,
        )
        self.linear = torch.nn.Linear(2 * n, 2 * n, bias=False)
        if learnable:
            self.linear.weight.data = fft_matrix + 1 / (np.sqrt(self.n) * 16) * torch.randn_like(fft_matrix)
        else:
            self.linear.weight.data = fft_matrix
        
        self.linear.weight.requires_grad = learnable

    def forward(self, x):
        xt = torch.transpose(x, self.dim, -1)
        x_real = xt[..., 0, :, :]
        x_imag = xt[..., 1, :, :]
        x_vec = torch.cat([x_real, x_imag], dim=-1)
        fft_vec = self.linear(x_vec)
        fft_real = fft_vec[..., : self.n]
        fft_imag = fft_vec[..., self.n :]
        return torch.transpose(
            torch.stack([fft_real, fft_imag], dim=-3), -1, self.dim
        )


class LearnableFourier2D(torch.nn.Module):
    """ Learnable 2D discrete Fourier transform.

    Implements a complex operator C^(n1, n2) -> C^(n1, n2), which is learnable
    but initialized as the Fourier transform. Operates along the last two
    dimensions of inputs with more axis.


    Parameters
    ----------
    n : tuple of int
        Dimensions of the domain and range of the operator.
    inverse : bool, optional
        Use the discrete inverse Fourier transform as initialization instead.
        (Default False)
    learnable : bool, optional
        Make operator learnable. Otherwise it will be kept fixed as the
        initialization. (Default True)

    """

    def __init__(self, n, inverse=False, learnable=True):
        super(LearnableFourier2D, self).__init__()
        self.inverse = inverse
        self.linear1 = LearnableFourier1D(
            n[0], dim=-2, inverse=inverse, learnable=learnable
        )
        self.linear2 = LearnableFourier1D(
            n[1], dim=-1, inverse=inverse, learnable=learnable
        )

    def forward(self, x):
        x = self.linear1(self.linear2(x))
        return x

class LearnableInverterFourier(torch.nn.Module):
    """ Learnable inversion of subsampled discrete Fourier transform.

    The zero-filling (transpose of the subsampling operator) is fixed.
    The inversion is learnable and initialized as a 2D inverse Fourier
    transform, realized as Kroneckers of 1D Fourier inversions.

    Implements a complex operator C^m -> C^(n1, n2).


    Parameters
    ----------
    n : tuple of int
        Dimensions of the range of the operator.
    mask : torch.Tensor
        The subsampling mask. Determines m.

    """

    def __init__(self, n, mask, learnable=True):
        super(LearnableInverterFourier, self).__init__()
        self.n = n
        self.mask = mask[0, 0, :, :].bool()
        self.learnable_ifft = LearnableFourier2D(
            n, inverse=True, learnable=learnable
        )

    def forward(self, y):
        masked_fft = torch.zeros(
            *y.shape[:-1], self.n[0] * self.n[1], device=y.device
        )
        masked_fft[..., im2vec(self.mask)] = y
        masked_fft = vec2im(masked_fft, self.n)
        return self.learnable_ifft(masked_fft)


##############################################################################
###                     Hadamard operators                                 ###
##############################################################################

class Hadamard(LinearOperator):
    """ 2D discrete Hadamard transform.

    Implements the linear operator R^(n1, n2) -> R^m
    appling the (subsampled) Hadamard transform.
    The inverse is the same as adjoint.


    Parameters
    ----------
    mask : torch.Tensor
        The subsampling mask for the Hadamard transform.

    """

    def __init__(self, mask, device=None):
        m = mask.nonzero().shape[0]
        n = mask.shape[-2:]
        super().__init__(m, n)
        self.mask = mask[0, 0, :, :].bool()
        self.H_mat = LearnableHadamard2D(n, learnable=False, device=device)

    def dot(self, x):
        """ Subsampled Hadamard transform. """
        x = self.H_mat(x)
        return im2vec(x)[..., im2vec(self.mask)]

    def adj(self, y):
        """ Adjoint is the zero-filled inverse Hadamard transform. """
        masked_had = torch.zeros(
            *y.shape[:-1], self.n[0] * self.n[1], device=y.device
        )
        masked_had[..., im2vec(self.mask)] = y
        x = vec2im(masked_had, self.n)       
        x = self.H_mat(x)
        return x

    def inv(self, y):
        """ Pseudo-inverse a.k.a. zero-filled WHT. """
        return self.adj(y)



class LearnableHadamard1D(torch.nn.Module):
    """ Learnable 1D discrete Hadamard transform.

    Implements a linear operator R^n -> R^n, which is learnable but
    initialized as the Hadamard transform.


    Parameters
    ----------
    n : int
        Dimension of the domain and range of the operator.
    dim : int, optional
        Apply the 1D operator along specified axis for inputs with multiple
        axis. (Default is last axis)
    learnable : bool, optional
        Make operator learnable. Otherwise it will be kept fixed as the
        initialization. (Default True)

    """

    def __init__(self, n, dim=-1, learnable=True, device=None):
        super(LearnableHadamard1D, self).__init__()

        self.n = n
        self.dim = dim
        self.device = device
        had_matrix = torch.from_numpy(hadamard_paley(n)/np.sqrt(n)).float().to(device)



        self.linear = torch.nn.Linear(n,n, bias=False, dtype=torch.float32, device=self.device)
        
        if learnable:
            self.linear.weight.data = had_matrix + 1 / (np.sqrt(self.n) * 16) * torch.randn_like(had_matrix)
        else:
            self.linear.weight.data = had_matrix
        
        self.linear.weight.requires_grad = learnable

    def forward(self, x):
        xt = torch.transpose(x, self.dim, -1)
        had_vec = self.linear(xt)
        return torch.transpose(
            had_vec, -1, self.dim
        )

class LearnableHadamard2D(torch.nn.Module):
    """ Learnable 2D discrete Hadamard transform.

    Implements a linear operator R^(n1, n2) -> R^(n1, n2), which is learnable
    but initialized as the Hadamard transform. Operates along the last two
    dimensions of inputs with more axis.


    Parameters
    ----------
    n : tuple of int
        Dimensions of the domain and range of the operator.
    learnable : bool, optional
        Make operator learnable. Otherwise it will be kept fixed as the
        initialization. (Default True)

    """

    def __init__(self, n, learnable=True, device=None):
        self.device = device
        super(LearnableHadamard2D, self).__init__()
        self.linear1 = LearnableHadamard1D(
            n[0], dim=-2, learnable=learnable, device=self.device
        )
        self.linear2 = LearnableHadamard1D(
            n[1], dim=-1, learnable=learnable, device=self.device
        )

    def forward(self, x):
        x = self.linear1(self.linear2(x))
        return x

class LearnableInverterHadamard(torch.nn.Module):
    """ Learnable inversion of subsampled discrete Hadamard transform.

    The zero-filling (transpose of the subsampling operator) is fixed.
    The inversion is learnable and initialized as a 2D Hadamard 
    transform, realized as Kroneckers of 1D Hadamard inversions.

    Implements a operator R^m -> R^(n1, n2).


    Parameters
    ----------
    n : tuple of int
        Dimensions of the range of the operator.
    mask : torch.Tensor
        The subsampling mask. Determines m.

    """

    def __init__(self, n, mask, learnable=True):
        super(LearnableInverterHadamard, self).__init__()
        self.n = n
        self.mask = mask[0, 0, :, :].bool()
        self.learnable_had = LearnableHadamard2D(
            n, learnable=learnable
        )

    def forward(self, y):
        masked_had = torch.zeros(
            *y.shape[:-1], self.n[0] * self.n[1], device=y.device
        )
        masked_had[..., im2vec(self.mask)] = y
        masked_had = vec2im(masked_had, self.n)
        return self.learnable_had(masked_had)
