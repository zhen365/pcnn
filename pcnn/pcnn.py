import numpy as np
import scipy.signal as sg

class PCNN:
    def __init__(self, img, af=0.1, al=1.0, ae=0.1, vf=0.5, vl=0.2, ve=20., beta=0.1) -> None:
        self.S = img
        self.F, self.L, self.U, self.Y, self.E = [np.zeros_like(img) for _ in range(5)]
        self.af = af
        self.al = al
        self.ae = ae
        self.vf = vf
        self.vl = vl
        self.ve = ve
        self.beta = beta
        self.M = np.array([[0.5, 1., 0.5], [1., 0., 1.], [0.5, 1., 0.5]])
        self.W = np.array([[0.5, 1., 0.5], [1., 0., 1.], [0.5, 1., 0.5]])
    
    def iterate(self, steps):
        for _ in range(steps):
            self.F = np.exp(-self.af)*self.F + self.vf*sg.convolve2d(self.Y, self.M, mode="same") + self.S
            self.L = np.exp(-self.al)*self.L + self.vl*sg.convolve2d(self.Y, self.W, mode="same")
            self.U = self.F*(1 + self.beta*self.L)
            self.Y = np.where(self.U >= self.E, 1., 0.)
            self.E = np.exp(-self.ae)*self.E + self.ve*self.Y
        return self.Y

class ICM:
    def __init__(self, img, f=0.9, g=0.8, h=20) -> None:
        self.S = img
        self.F, self.E, self.Y = [np.zeros_like(img) for _ in range(3)]
        self.f = f
        self.g = g
        self.h = h
        self.M = np.array([[0.5, 1., 0.5], [1., 0., 1.], [0.5, 1., 0.5]])
    
    def iterate(self, steps):
        for _ in range(steps):
            self.F = self.f*self.F + sg.convolve2d(self.Y, self.M, mode="same") + self.S
            self.Y = np.where(self.F >= self.E, 1., 0.)
            self.E = self.g*self.E + self.h*self.Y
        return self.Y

class SPCNN:
    def __init__(self, img, otsu_ret) -> None:
        self.S = img / 255.
        self.otsu_ret = otsu_ret
        self.initialize()
        self.L, self.U, self.Y, self.E, self.T = [np.zeros_like(img) for _ in range(5)]
        self.M = np.array([[0.5, 1., 0.5], [1., 0., 1.], [0.5, 1., 0.5]])
        self.info()

    def initialize(self):
        Shist = self.otsu_ret/ 255.
        theta = np.std(self.S)
        Smax = np.max(self.S)
        self.af = np.log(1/theta)
        self.vl = 1
        self.beta = beta = (Smax/Shist - 1)/(6 * self.vl)
        self.ve = np.exp(-self.af) + 1 + 6 * beta * self.vl
        M3 = (1 - np.exp(-(3 * self.af)))/(1 - np.exp(-self.af)) + 6 * beta * self.vl * np.exp(-self.af)
        self.ae = np.log(self.ve/(Shist*M3))

    def info(self):
        print(f"SPCNN: [af: {self.af}, vl: {self.vl}, beta: {self.beta}, ve: {self.ve}, ae: {self.ae}]")

    def iterate(self, steps):
        for i in range(steps):
            self.F = self.S
            self.L = self.vl*sg.convolve2d(self.Y, self.M, mode="same")
            self.U = np.exp(-self.af) * self.U + self.F*(1 + self.beta*self.L)
            self.Y = np.where(self.U >= self.E, 1, 0)
            self.E = np.exp(-self.ae)*self.E + self.ve*self.Y
            self.T = self.T + self.Y
        return self.Y
    