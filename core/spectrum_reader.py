from glob import glob
from os.path import join as make_path
from re import compile
import numpy as np
from numpy import pi, array, zeros, float64, mean, where
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
        self.__rad_per_deg = pi / 180

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
        steps *= self.__rad_per_deg * self.__deg_per_micron * self.__micron_per_step

        return steps

    @staticmethod
    def __check(steps, lambdas, spectrum):
        assert (steps.shape[0] == spectrum.shape[0]), 'Different steps and spectra dimensions'
        assert (lambdas.shape[0] == spectrum.shape[1]), 'Different lambdas and spectra dimensions'

    @jit(nopython=True)
    def linear_approximation_real(x, x1, y1, x2, y2):
        """Linear approximation for float arguments"""

        return (y1 - y2) / (x1 - x2) * x + (y2 * x1 - x2 * y1) / (x1 - x2)

    def __make_uniform_along_angle(self, angles, spectrum):
        pass

    def __plot_fas(self, spectrum):
        plt.figure(figsize=(15, 10))
        plt.contourf(spectrum, cmap='gray', levels=100)

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

        # make angles and spectrum uniform


        self.__check(steps, lambdas, spectrum)

        print(steps)

        self.__plot_fas(spectrum)






