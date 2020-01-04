from glob import glob
from os import mkdir
from os.path import join as make_path
import datetime
from re import compile
import numpy as np
from numpy import zeros, float64, where, log10
from math import isclose
from abc import ABCMeta, abstractmethod


class BaseProcessor(metaclass=ABCMeta):
    def __init__(self, experimental_data_dir, **kwargs):

        self._dir = experimental_data_dir
        self._dirname = self._dir.split('\\')[-1]
        self._res_dir = 'results_%s_%s' % (self._dirname, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        mkdir(self._res_dir)

        self._regex_expr = r'\d\.\d+|\d+\t[-+]?\d+\.\d+|\d+\n'
        self._sigma_lambda = kwargs.get('sigma_lambda', 10.0)  # [nm]
        self._log_power = kwargs.get('log_power', -2)

    @abstractmethod
    def _process(self):
        """"""

    def _logarithm(self, spectrum):
        maximum = np.max(spectrum)
        lowest_levels = maximum * 10**self._log_power
        spectrum[where(spectrum < lowest_levels)] = lowest_levels

        return log10(spectrum / maximum)

    def _get_files(self):
        files = []
        for file in glob(make_path(self._dir, '*.dat')):
            files.append(file.replace('\\', '/'))

        return files

    def _get_proper_lines(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()
        regex = compile(self._regex_expr)
        lines = list(filter(regex.search, lines))

        n = len(lines)
        if n < 10:
            raise Exception('Small number of lambdas!')

        return lines, n

    def _get_lambdas(self, file):
        lines, n = self._get_proper_lines(file)
        lambdas = zeros(shape=(n,), dtype=float64)
        for i, line in enumerate(lines):
            lambdas[i] = float(line.split('\t')[0])

        for i in range(1, len(lambdas)-1, 1):
            if not isclose(lambdas[i] - lambdas[i-1], lambdas[i + 1] - lambdas[i], rel_tol=0.05):
                raise Exception('Step along lambdas is not constant!')

        return lambdas

    def _get_spectrum(self, file):
        lines, n = self._get_proper_lines(file)
        spectrum = zeros(shape=(n,), dtype=float64)

        for i, line in enumerate(lines):
            spectrum[i] = float(line[:-1].split('\t')[-1])

        return spectrum
