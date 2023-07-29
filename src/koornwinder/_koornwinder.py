import numpy as np
import json

import importlib.resources
with importlib.resources.path("koornwinder", "koornwinder.json") as _f:
  _data_str = open(_f).read()

class Koornwinder:
    """
    The koornwinder polynomial is a bivariate orthogonormal polynomial defined on the triangle on the simplex (0,0) - (1,0) - (0,1).

    Using the precomputed data in koornwinder.json, this class provides several functionalities for 
    - quadrature
    - interpolation
    - evaluating polynomial
    - computing polynomial expansion coefficients
    - spectral differentiation
    on the precomputed discretization nodes with the precomputed quadrature weights. 

    It only supports to koornwinder polynomials of 0 <= degree <= 20.

    This class is a python implementation of the original fortran code at
    https://github.com/fastalgorithms/fmm3dbie/blob/master/src/tria_routs/koornexps.f90
    """

    _data = json.loads(_data_str)
    norder: int       # the maximum degree of the koornwinder polynomials
    uvs: np.ndarray   # discretization nodes
    whts: np.ndarray  # quadrature weights
    vmat: np.ndarray  # the matrix that maps the coefficients to the values at the nodes
    umat: np.ndarray  # the matrix that maps the values at the nodes to the coefficients

    def __init__(self, norder) -> None:
        uvs, whts = Koornwinder.nodes_whts(norder)
        vmat = Koornwinder.coef2val(norder, uvs)
        umat = np.linalg.inv(vmat)

        self.norder = norder
        self.uvs = uvs
        self.whts = whts
        self.vmat = vmat
        self.umat = umat

    @staticmethod
    def npols(norder):
        return int((norder + 1) * (norder + 2) / 2)

    @staticmethod
    def whts(norder):
        assert 0 <= norder <= 20, "norder must be between 0 and 20"
        return np.array(Koornwinder._data[str(norder)]['w'])

    @staticmethod
    def nodes(norder):
        assert 0 <= norder <= 20, "norder must be between 0 and 20"
        u = Koornwinder._data[str(norder)]['u']
        v = Koornwinder._data[str(norder)]['v']
        return np.array([u, v])

    @staticmethod
    def nodes_whts(norder):
        uvs = Koornwinder.nodes(norder)
        whts = Koornwinder.whts(norder)
        return uvs, whts

    @staticmethod
    def in_simplex(uv):
        return np.all(0 <= uv) and np.all(uv[0] + uv[1] <= 1)

    @staticmethod
    def pols(uv,nmax):
        """
        Evaluate orthogonal polynomials on the simplex at uv = (u,v) up to degree nmax
        """
        return Koornwinder.pols_and_ders(uv, nmax, ders=False)
    
    @staticmethod
    def pols_and_ders(uv, nmax, ders=True):
        """
        Evaluate orthogonal polynomials and derivatives on the simplex at uv = (u,v) up to degree nmax
        """

        assert Koornwinder.in_simplex(uv), "uv must be in the simplex"
        assert 0 <= nmax <= 1000, "nmax must be between 0 and 1000"

        npols = Koornwinder.npols(nmax)
        npts = uv.shape[1]
        pols = np.empty((npols, npts))

        # Build the legpols
        legpols = np.empty((nmax + 2, npts))
        u = uv[0]
        v = uv[1]
        z = 2 * u + v - 1
        y = 1 - v
        legpols[0] = 1
        legpols[1] = z
        for k in range(1, nmax + 1):
            legpols[k + 1] = ((2 * k + 1) * z * legpols[k] - k * legpols[k - 1] * y**2) / (k + 1)

        # Build the jacpols

        x = 1 - 2 * v
        jacpols = np.zeros((nmax + 1, nmax + 1, npts))

        for k in range(nmax + 1):
            beta = 2 * k + 1
            jacpols[0, k] = 1
            jacpols[1, k] = (-beta + (2 + beta) * x) / 2

            for n in range(1, nmax - k):
                an = (2 * n + beta + 1) * (2 * n + beta + 2) / 2 / (n + 1) / (n + beta + 1)
                bn = (-(beta**2)) * (2 * n + beta + 1) / 2 / (n + 1) / (n + beta + 1) / (2 * n + beta)
                cn = n * (n + beta) * (2 * n + beta + 2) / (n + 1) / (n + beta + 1) / (2 * n + beta)
                jacpols[n + 1, k] = (an * x + bn) * jacpols[n, k] - cn * jacpols[n - 1, k]

        # Build the koornwinder pols
        iii = 0
        for n in range(nmax + 1):
            for k in range(n + 1):
                sc = np.sqrt(1 / (2 * k + 1) / (2 * n + 2))
                iii += 1
                pols[iii - 1] = legpols[k] * jacpols[n - k, k] / sc

        if not ders:
            return pols

        ### computing ders ###
        ders = dict()
        ders['u'] = np.empty((npols,npts))
        ders['v'] = np.empty((npols,npts))        

        # first run for legu, legv
        legu = np.empty((nmax + 2, npts))
        legv = np.empty((nmax + 2, npts))

        legu[0] = 0
        legu[1] = 2

        legv[0] = 0
        legv[1] = 1

        for k in range(1, nmax + 1):
            legu[k + 1] = ((2 * k + 1) * (2 * legpols[k] + z * legu[k]) - k * legu[k - 1] * y * y) / (k + 1)
            legv[k + 1] = (
                (2 * k + 1) * (legpols[k] + z * legv[k]) - k * (legv[k - 1] * y * y - 2 * legpols[k - 1] * y)
            ) / (k + 1)

        # second run for jacv
        jacv = np.zeros((nmax + 1, nmax + 1, npts))
        for k in range(nmax + 1):
            beta = 2 * k + 1
            jacv[0, k] = 0
            jacv[1, k] = -(2 + beta)

            for n in range(1, nmax - k):
                an = (2 * n + beta + 1) * (2 * n + beta + 2) / 2 / (n + 1) / (n + beta + 1)
                bn = (-(beta**2)) * (2 * n + beta + 1) / 2 / (n + 1) / (n + beta + 1) / (2 * n + beta)
                cn = n * (n + beta) * (2 * n + beta + 2) / (n + 1) / (n + beta + 1) / (2 * n + beta)
                jacv[n + 1, k] = -2 * an * jacpols[n, k] + (an * x + bn) * jacv[n, k] - cn * jacv[n - 1, k]

        # now assemble the koornwinder ders
        iii = 0
        for n in range(nmax + 1):
            for k in range(n + 1):
                sc = np.sqrt(1 / (2 * k + 1) / (2 * n + 2))
                iii += 1
                ders['u'][iii - 1] = legu[k] * jacpols[n - k, k] / sc
                ders['v'][iii - 1] = (legv[k] * jacpols[n - k, k] + legpols[k] * jacv[n - k, k]) / sc

        return pols, ders

    @staticmethod
    def val2coef(nmax, uv=None):
        """
        Given uvs, arbitrary npols of points on the simplex, return a matrix that
        maps the values at those points to the coefficients of koornwinder polynomials expansion of degree <= nmax.

        the distribution of uvs would affect the condition number of the matrix.

        if uvs=None, then the quadrature nodes are used.
        """

        npols = Koornwinder.npols(nmax)

        if uv is None:
            uv = Koornwinder.nodes(nmax)
        else:
            assert Koornwinder.in_simplex(uv), "uvs must be in the simplex"
            assert uv.shape[0] == 2 and uv.shape[1] == npols, "uvs must be a 2 by npts array" 
        
        return np.linalg.inv(Koornwinder.coef2val(nmax, uv))

    @staticmethod
    def coef2val(nmax, uv=None):
        npols = Koornwinder.npols(nmax)
        if uv is None:
            uv = Koornwinder.nodes(nmax)
        else:
            assert Koornwinder.in_simplex(uv), "uvs must be in the simplex"
            assert uv.shape[0] == 2, "uvs must be an array of 2d points" 
        
        return Koornwinder.pols(uv, nmax).T

    @staticmethod
    def upsample(korder, norder):
        # the upsampling matrix for korder < norder with shape = (npols, kpols)

        # uvk -> coefk
        val2coef = Koornwinder.val2coef(korder)

        # coefk -> uvn
        uvn = Koornwinder.nodes(norder)
        coef2val = Koornwinder.coef2val(korder, uvn)

        return coef2val @ val2coef

    @staticmethod
    def eval(coef, uv=None, ders=0):
        """
        ders is an integer determining the order of the derivative to be evaluated.
        """

        assert ders >= 0, "ders must be >= 0"

        # determine the order of the polynomial
        nmax = 0
        while Koornwinder.npols(nmax) < coef.shape[0]:
            nmax += 1
        npols = Koornwinder.npols(nmax)

        # extending coef if necessary
        if coef.shape[0] < npols:
            coef = np.concatenate((coef, np.zeros(npols - coef.shape[0])))

        if uv is None:
            uv = Koornwinder.nodes(nmax)
        else:
            assert Koornwinder.in_simplex(uv), "uv must be in the simplex"

        if ders == 0:
            return Koornwinder.coef2val(nmax, uv) @ coef

        # 1st order derivative coef2val matrix. 
        pols, ders = Koornwinder.pols_and_ders(uv, nmax, ders=True)
        val = pols.T @ coef

        # 1st order derivative val. 
        der = dict()
        der['u'] = ders['u'].T @ coef
        der['v'] = ders['v'].T @ coef

        if ders == 1:
            return val, der

        # higher order derivative
        # building the spectral differentiation matrix.
        # polynomial values -> polynomial coefficients -> values of derivative of polynomial
        val2coef = np.linalg.inv(pols.T)
        du_mat = ders['u'].T @ val2coef
        dv_mat = ders['v'].T @ val2coef

        # 2nd order derivative
        der['uu'] = du_mat @ der['u']
        der['uv'] = du_mat @ der['v']
        der['vv'] = dv_mat @ der['v']

        if ders == 2:
            return val, der

        raise NotImplementedError("ders > 2 is not implemented")
        # it's easy to implement though. 

    @staticmethod
    def vioreanu_simplex_quad(norder):
        uvs, whts = Koornwinder.nodes_whts(norder)
        vmat = Koornwinder.coef2val(norder, uvs)
        umat = np.linalg.inv(vmat)

        return uvs, whts, umat, vmat
