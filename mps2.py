import numpy as np

# CONST
p1, p3 = (1+1.j/np.sqrt(3))/4, 1/2
p2 = 2*p1
p5, p4 = np.conj(p1), np.conj(p2)
x0, x1 = -2**(1/3)/(2-2**(1/3)), 1/(2-2**(1/3))

class MPS:
    """ MPS for calculation of the ground state in the antiferromagnetic Heisenberg model. """

    def __init__(self, n, z, Mmax=10):
        """ Initialization of MPS state: Neel state. """

        def create_simple_MPS(n):
            s0, s1 = [np.array([1]), np.array([0])], [np.array([0]), np.array([1])]
            return [s0 if i%2==0 else s1 for i in range(n)], [np.array([1]) for _ in range(n+1)]    # Bs, langdas

        Bs, langdas = create_simple_MPS(n)
        self.Bs = Bs
        self.langdas = langdas
        self.n = n
        self.e = np.exp(-z)
        self.ech = np.exp(z)*np.cosh(2*z)
        self.esh = np.exp(z)*np.sinh(2*z)
        self.betas = []
        self.Es = []
        self.Mmax = Mmax
        self.error = 0
        self.errors = []
        self.norm = 1.
        self.gain = 0.

    def set_parameters(self, z):
        """ Set new parameters of the two gate unitary operator: exp(-z), exp(z)*cosh(z), exp(z)*sinh(z). """
        self.e = np.exp(-z)
        self.ech = np.exp(z)*np.cosh(2*z)
        self.esh = np.exp(z)*np.sinh(2*z)

    def renormalize(self):
        """ Renormalization of diagonal lambda matrices and increment of gain, which is used for energy calculation.
        This method is crucial for large systems (n) and inverse temperatures (beta). """
        for i in range(1, self.n):
            nrm = np.sqrt(np.sum(self.langdas[i]**2))
            self.langdas[i] = self.langdas[i]/nrm
            self.gain += np.log(nrm)

    def local_interaction(self, j):
        """ Basic building block of the algorithm. """

        Bj0, Bj1 = self.Bs[j], self.Bs[j+1]
        langda0, langda1, langda2 = self.langdas[j][:, np.newaxis], self.langdas[j+1], self.langdas[j+2]
        langda02 = langda0*langda2

        B00 = langda02 * (self.e*np.matmul(Bj0[0]*langda1, Bj1[0]))
        B01 = langda02 * (self.ech*np.matmul(Bj0[0]*langda1, Bj1[1]) - self.esh*np.matmul(Bj0[1]*langda1, Bj1[0]))
        B10 = langda02 * (self.ech*np.matmul(Bj0[1]*langda1, Bj1[0]) - self.esh*np.matmul(Bj0[0]*langda1, Bj1[1]))
        B11 = langda02 * (self.e*np.matmul(Bj0[1]*langda1, Bj1[1]))

        rows = 2*B00.shape[0]
        Q = np.dstack([np.hstack([B00, B10]).reshape(rows,-1), np.hstack([B01, B11]).reshape(rows,-1)]).reshape(rows,-1)

        u, langda, vh = np.linalg.svd(Q, full_matrices=False)
        Mj = min(langda.size, self.Mmax)
        self.error += np.sum(np.square(langda[Mj:]))

        self.Bs[j] = [np.array(u[0::2, :Mj]/langda0), np.array(u[1::2, :Mj]/langda0)]
        self.langdas[j+1] = langda[:Mj]
        self.Bs[j+1] = [np.array(vh[:Mj, 0::2]/langda2), np.array(vh[:Mj, 1::2]/langda2)]

    def step_even(self):
        """ Even interactions in the Suzuki-Trotter scheme. """
        self.renormalize()
        for j in range(self.n//2):
            self.local_interaction(2*j)

    def step_odd(self):
        """ Odd interactions in the Suzuki-Trotter scheme. """
        self.renormalize()
        for j in range(self.n//2-1):
            self.local_interaction(2*j+1)

    def stepS2(self, z):
        """ Symmetric S2 Split-step scheme (Trotter) """
        self.set_parameters(z/2); self.step_even()
        self.set_parameters(z); self.step_odd()
        self.set_parameters(z/2); self.step_even()

    def stepS3(self, z):
        """ S3 Split-step scheme """
        self.set_parameters(p1*z); self.step_even()
        self.set_parameters(p2*z); self.step_odd()
        self.set_parameters(p3*z); self.step_even()
        self.set_parameters(p4*z); self.step_odd()
        self.set_parameters(p5*z); self.step_even()

    def stepS4(self, z):
        """ S4 Split-step scheme """
        self.set_parameters(x1/2*z); self.step_even()
        self.set_parameters(x1*z); self.step_odd()
        self.set_parameters((x1+x0)/2*z); self.step_even()
        self.set_parameters(x0*z); self.step_odd()
        self.set_parameters((x1+x0)/2*z); self.step_even()
        self.set_parameters(x1*z); self.step_odd()
        self.set_parameters(x1/2*z); self.step_even()

    def stepS3S3(self, z):
        """ S3\bar{S}3 Split-step scheme (Symmetrized S3) """
        self.set_parameters(p1/2*z); self.step_even()
        self.set_parameters(p2/2*z); self.step_odd()
        self.set_parameters(p3/2*z); self.step_even()
        self.set_parameters(p4/2*z); self.step_odd()
        self.set_parameters(p5*z); self.step_even()
        self.set_parameters(p4/2*z); self.step_odd()
        self.set_parameters(p3/2*z); self.step_even()
        self.set_parameters(p2/2*z); self.step_odd()
        self.set_parameters(p1/2*z); self.step_even()

    def propagate(self, step_scheme='S2', dbeta=0.01, bsteps=100):
        """ Propagation with chosen Split-step scheme and energy calculation. """

        def err_func(*arg):
            msg = 'Scheme '+step_scheme+' is not supported. Supported schemes are: '+\
                  ', '.join([s[4:] for s in dir(self) if s[:4]=='step' and not (s[4:]=='_even' or s[4:]=='_odd')])+'.'
            raise ValueError(msg)

        S = getattr(self, 'step'+step_scheme, err_func)

        for beta in np.arange(dbeta, (bsteps+1)*dbeta, dbeta):
            S(dbeta)
            self.calc_energy(beta)

    def calc_energy(self, beta):
        """ Energy calculation and appending the result to Es attribute. """
        E = np.array([[1]])
        for k in range(self.n):
            E = np.matmul(E, self.langdas[k][:, np.newaxis] * self.Bs[k][k%2])

        self.betas.append(beta)
        self.Es.append((-np.log(E[0][0])-self.gain) / beta)
        self.errors.append(self.error)

    def calc_mps_norm(self):
        """ Calculation of the MPS norm squared. """
        nrm = 1
        for i in range(len(self.Bs)):
            langda0 = self.langdas[i][:, np.newaxis]
            Bj0 = langda0*self.Bs[i][0]
            Bj1 = langda0*self.Bs[i][1]
            nrm = np.dot(nrm, np.kron(Bj0, Bj0) + np.kron(Bj1, Bj1))

        self.norm = nrm[0][0]

    @staticmethod
    def calc_spin_correlation(mps):
        """ Calculation of the spin-spin correlation matrix. (Naive) """

        def spin_corr(Bs, langdas, j, k):
            corr = 1
            for i in range(len(Bs)):
                langda0 = langdas[i][:, np.newaxis]
                Bj0 = langda0*Bs[i][0]
                Bj1 = langda0*Bs[i][1]

                if i == j or i == k:
                    corr = np.dot(corr, np.kron(Bj0, Bj0) - np.kron(Bj1, Bj1))
                else:
                    corr = np.dot(corr, np.kron(Bj0, Bj0) + np.kron(Bj1, Bj1))

            return corr[0][0]

        n = mps.n
        mat = np.empty((n, n))
        mps.calc_mps_norm()

        for j in range(n):
            mat[j, j] = 1
            for k in range(j+1, n):
                ss = spin_corr(mps.Bs, mps.langdas, j, k) / mps.norm
                mat[j, k], mat[k, j] = ss, ss

        return mat

    @staticmethod
    def calc_spin_correlation2(mps):
        """ Calculation of the spin-spin correlation matrix. (Advance) """

        n = mps.n
        mat = np.empty((n, n))
        mps.calc_mps_norm()

        L = 1
        for j in range(n):
            mat[j, j] = 1

            if j > 0:
                langda0 = mps.langdas[j-1][:, np.newaxis]
                Bj0 = langda0 * mps.Bs[j-1][0]
                Bj1 = langda0 * mps.Bs[j-1][1]
                L = np.dot(L, np.kron(Bj0, Bj0) + np.kron(Bj1, Bj1))

            R = 1
            for k in range(j+1, n)[::-1]:
                if k < n-1:
                    langda0 = mps.langdas[k+1][:, np.newaxis]
                    Bj0 = langda0 * mps.Bs[k+1][0]
                    Bj1 = langda0 * mps.Bs[k+1][1]
                    R = np.dot(np.kron(Bj0, Bj0) + np.kron(Bj1, Bj1), R)

                corr = L
                for i in range(j, k+1):
                    langda0 = mps.langdas[i][:, np.newaxis]
                    Bj0 = langda0 * mps.Bs[i][0]
                    Bj1 = langda0 * mps.Bs[i][1]

                    if i == j or i == k:
                        corr = np.dot(corr, np.kron(Bj0, Bj0) - np.kron(Bj1, Bj1))
                    else:
                        corr = np.dot(corr, np.kron(Bj0, Bj0) + np.kron(Bj1, Bj1))

                corr = np.dot(corr, R)

                corr = corr[0][0]/mps.norm
                mat[j, k], mat[k, j] = corr, corr

        return mat