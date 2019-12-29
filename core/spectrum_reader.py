from glob import glob
from os.path import join as make_path
from re import compile
import numpy as np
from numpy import pi, array, zeros, float64, mean, where, deg2rad, inf
from math import isclose
from collections import OrderedDict
from matplotlib import pyplot as plt
from pylab import contourf
from numba import jit


class SpectrumReader:
    def __init__(self, spectrum_dir, **kwargs):

        self.__dir = spectrum_dir
        self.__dirname = self.__dir.split('\\')[-1]

        self.__regex_expr = r'\d\.\d+|\d+\t[-+]?\d+\.\d+|\d+\n'
        self.__micron_per_step = kwargs.get('micron_per_step', 10)
        self.__deg_per_micron = kwargs.get('deg_per_micron', 5 / 3500)

        self.__process()

    def __get_files(self):
        files = []
        for file in glob(make_path(self.__dir, '*.dat')):
            files.append(file.replace('\\', '/'))

        return files

    def __get_proper_lines(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()
        regex = compile(self.__regex_expr)
        lines = list(filter(regex.search, lines))

        n = len(lines)
        if n < 10:
            raise Exception('Small number of lambdas!')

        return lines, n

    def __get_lambdas(self, file):
        lines, n = self.__get_proper_lines(file)
        lambdas = zeros(shape=(n,), dtype=float64)
        for i, line in enumerate(lines):
            lmbda = float(line.split('\t')[0])
            lambdas[i] = lmbda

        for i in range(1, len(lambdas)-1, 1):
            if not isclose(lambdas[i] - lambdas[i-1], lambdas[i + 1] - lambdas[i], rel_tol=0.05):
                raise Exception('Step along lambdas is not constant!')

        return lambdas

    def __get_spectrum(self, file):
        lines, n = self.__get_proper_lines(file)
        spectrum = zeros(shape=(n,), dtype=float64)

        for i, line in enumerate(lines):
            spectrum[i] = float(line[:-1].split('\t')[-1])

        return spectrum

    def __get_data(self, files):
        data = {}
        for file in files:
            step = int(''.join(filter(str.isdigit, (file.split('\\')[-1]).split('.')[0])))
            spectrum = self.__get_spectrum(file)
            if step in data.keys():
                data[step].append(spectrum)
            else:
                data[step] = [spectrum]

        return OrderedDict(sorted([(step, data[step]) for step in data.keys()]))

    @staticmethod
    def __transform_data(data):
        transformed = OrderedDict()
        for step in data:
            averaged_arr = mean(data[step], axis=0)
            averaged_arr[where(averaged_arr < 0)] = 0.0
            transformed[step] = averaged_arr

        return transformed

    def __steps2angles(self, steps):
        steps -= np.min(steps)
        steps *= deg2rad(self.__deg_per_micron * self.__micron_per_step)

        return steps

    @staticmethod
    def __check(steps, lambdas, spectrum):
        assert (steps.shape[0] == spectrum.shape[0]), 'Different steps and spectra dimensions'
        assert (lambdas.shape[0] == spectrum.shape[1]), 'Different lambdas and spectra dimensions'

    @staticmethod
    @jit(nopython=True)
    def __make_uniform_along_angle(angles, spectrum):
        n_angles, n_lambdas = spectrum.shape
        dangle_uniform = (angles[-1] - angles[0]) / n_angles
        angles_uniform = [angles[0] + k * dangle_uniform for k in range(n_angles)]

        spectrum_uniform = zeros(shape=spectrum.shape)

        prev, nxt = 0, 1
        for k in range(n_angles):
            for pos in range(n_angles):
                if angles[pos] >= angles_uniform[k]:
                    prev = max(0, pos - 1)
                    nxt = min(prev + 1, n_angles - 1)
                    break
            for s in range(n_lambdas):
                x = angles_uniform[k]
                x1, x2 = angles[prev], angles[nxt]
                y1, y2 = spectrum[prev, s], spectrum[nxt, s]
                spectrum_uniform[k, s] = (y1 - y2) / (x1 - x2) * x + (y2 * x1 - x2 * y1) / (x1 - x2)

        return angles_uniform, spectrum_uniform

    @staticmethod
    @jit(nopython=True)
    def __make_uniform_along_lambda(lambdas, spectrum):
        n_angles, n_lambdas = spectrum.shape

        dlambdas_uniform = (lambdas[-1] - lambdas[0]) / n_lambdas
        lambdas_uniform = [lambdas[0] + s * dlambdas_uniform for s in range(n_lambdas)]

        spectrum_uniform = zeros(shape=spectrum.shape)

        prev, nxt = 0, 1
        for s in range(n_lambdas):
            for pos in range(n_lambdas):
                if lambdas[pos] >= lambdas_uniform[s]:
                    prev = max(0, pos - 1)
                    nxt = min(prev + 1, n_lambdas)
                    break
            if prev == nxt:
                break
            for k in range(n_angles):
                x = lambdas_uniform[s]
                x1, x2 = lambdas[prev], lambdas[nxt]
                y1, y2 = spectrum[k, prev], spectrum[k, nxt]
                spectrum_uniform[k, s] = (y1 - y2) / (x1 - x2) * x + (y2 * x1 - x2 * y1) / (x1 - x2)

        return lambdas_uniform, spectrum_uniform

    def __plot_fas(self, angles, lambdas, spectrum):
        plt.figure(figsize=(15, 10))
        plt.contourf(spectrum, cmap='gray', levels=100)

        x_ticks_labels = ['1.2', '1.4', '1.6', '1.8', '2.0', '2.2', '2.4']

        dl = (lambdas[-1] - lambdas[0]) / len(lambdas)
        x_ticks = [int((float(e) * 10**3 - lambdas[0]) / dl) for e in x_ticks_labels]
        plt.xticks(x_ticks, x_ticks_labels, fontsize=20)

        # n_y = 7
        # dy = (angles[-1] - angles[0]) / n_y
        # y_ticks = [i * dy for i in range(n_y)]
        # plt.yticks(y_ticks, y_ticks, fontsize=20)

        plt.xlabel('$\mathbf{\lambda}$, nm', fontsize=30, fontweight='bold')
        plt.ylabel('$\mathbf{\\theta}$, rad', fontsize=30, fontweight='bold')

        plt.grid(linewidth=2, linestyle='dotted', color='gray', alpha=0.5)

        plt.savefig('fas_%s' % self.__dirname, bbox_inches='tight')
        plt.close()

    def __process(self):
        files = self.__get_files()
        if not files:
            raise Exception('No files detected!')

        # get data
        data = self.__get_data(files)
        data = self.__transform_data(data)

        # lambdas
        lambdas = self.__get_lambdas(files[0])

        # steps and angles
        steps = array(list(data.keys()), dtype=float64)
        angles = self.__steps2angles(steps)

        # spectrum
        spectrum = array(list(data.values()), dtype=float64)

        # check
        self.__check(steps, lambdas, spectrum)

        # make spectrum uniform along both axes
        angles, spectrum = self.__make_uniform_along_angle(angles, spectrum)
        lambdas, spectrum = self.__make_uniform_along_lambda(lambdas, spectrum)

        self.__plot_fas(angles, lambdas, spectrum)






