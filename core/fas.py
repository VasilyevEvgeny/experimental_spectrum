from collections import OrderedDict
from numba import jit
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import numpy as np
from os.path import join as make_path
from os import mkdir
from numpy import array, zeros, float64, mean, where, deg2rad, log10
from matplotlib import pyplot as plt

from .base import BaseProcessor


class ProcessorFAS(BaseProcessor):
    """Frequency-angular spectrum"""

    def __init__(self, experimental_data_dir, **kwargs):
        super().__init__(experimental_data_dir, **kwargs)

        self.__micron_per_step = kwargs.get('micron_per_step', 10)  # [micron / step]
        self.__deg_per_micron = kwargs.get('deg_per_micron', 5 / 3500)  # [deg / micron]

        self.__sigma_angle = kwargs.get('sigma_angle', 0)  # [rad]
        self.__steps_overlap = kwargs.get('steps_overlap', 4)  # []
        self.__lambda_dn = kwargs.get('lambda_dn', 50)  # []

        self._process()

    def __get_data(self, files):
        data = {}
        for file in files:
            step = int(''.join(filter(str.isdigit, (file.split('\\')[-1]).split('.')[0])))
            spectrum = self._get_spectrum(file)
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

    def __smooth_spectrum(self, dangle, dlambda, spectrum):
        n_sigma_angle = self.__sigma_angle / dangle
        n_sigma_lambda = self._sigma_lambda / dlambda

        return gaussian_filter(spectrum, sigma=(n_sigma_angle, n_sigma_lambda))

    def __cut_steps_overlap(self, angles, spectrum):
        angles = angles[self.__steps_overlap:]
        angles -= np.min(angles)
        return angles, spectrum[self.__steps_overlap:, :]

    @staticmethod
    def __reflect(angles, spectrum):
        angles = array([-e for e in list(angles)][::-1][:-1] + list(angles), dtype=float64)
        spectrum = array((list(spectrum)[::-1])[:-1] + list(spectrum), dtype=float64)

        return angles, spectrum

    def __plot(self, angles, lambdas, fas):

        ylabel = 'lg(S/S$\mathbf{_{max}}$)' if self._log_scale else 'S/S$\mathbf{_{max}}$'
        min_val, max_val = np.min(fas), np.max(fas)
        delta = 0.1 * (max_val - min_val)

        #
        # fas
        #

        fig, ax = plt.subplots(figsize=(15, 7))
        plot = plt.contourf(fas, cmap='gray', levels=100)

        x_ticks_labels = ['1.4', '1.6', '1.8', '2.0', '2.2', '2.4']
        dlambda = lambdas[1] - lambdas[0]
        x_ticks = [int((float(e) * 10**3 - lambdas[0]) / dlambda) for e in x_ticks_labels]
        plt.xticks(x_ticks, x_ticks_labels, fontsize=20)

        y_ticks_labels = ['-0.01', ' 0.00', '+0.01']
        dangle = angles[1] - angles[0]
        y_ticks = [int((float(e) + angles[-1]) / dangle) for e in y_ticks_labels]
        plt.yticks(y_ticks, y_ticks_labels, fontsize=20)

        plt.xlabel('$\mathbf{\lambda}$, $\mathbf{\mu}$m', fontsize=30, fontweight='bold')
        plt.ylabel('$\mathbf{\\theta}$, rad', fontsize=30, fontweight='bold')

        plt.grid(linewidth=2, linestyle='dotted', color='gray', alpha=0.5)

        n_ticks_colorbar_levels = 4
        dcb = (max_val - min_val) / n_ticks_colorbar_levels
        levels_ticks_colorbar = [min_val + i * dcb for i in range(n_ticks_colorbar_levels + 1)]

        colorbar = fig.colorbar(plot, ticks=levels_ticks_colorbar, orientation='vertical', aspect=10, pad=0.05)
        colorbar.set_label(ylabel, labelpad=-100, y=1.2, rotation=0, fontsize=30, fontweight='bold')
        ticks_cbar = ['%05.2f' % e if e != 0 else '00.00' for e in levels_ticks_colorbar]
        colorbar.ax.set_yticklabels(ticks_cbar)
        colorbar.ax.tick_params(labelsize=30)

        plt.savefig(make_path(self._current_res_dir, 'fas'), bbox_inches='tight')
        plt.close()

        #
        # frequency spectra
        #

        frequency_spectra_path = make_path(self._current_res_dir, 'frequency_spectra')
        mkdir(frequency_spectra_path)
        for i in tqdm(range(fas.shape[0] // 2 + 1), desc='%s->frequency_spectra' % self._current_res_dir):
            spectrum = fas[i, :]

            plt.figure(figsize=(20, 10))
            plt.plot(lambdas, spectrum, color='black', linewidth=5, linestyle='solid')

            plt.ylim([min_val - delta, max_val + delta])

            plt.xticks(fontsize=20, fontweight='bold')
            plt.yticks(fontsize=20, fontweight='bold')

            plt.xlabel('$\mathbf{\lambda}$, nm', fontsize=30, fontweight='bold')
            plt.ylabel(ylabel, fontsize=30, fontweight='bold')

            plt.grid(linewidth=2, linestyle='dotted', color='gray', alpha=0.5)

            plt.savefig(make_path(frequency_spectra_path, 'angle=%.5frad.png' % abs(angles[i])))
            plt.close()

        #
        # angular spectra
        #

        angular_spectra_path = make_path(self._current_res_dir, 'angular_spectra')
        mkdir(angular_spectra_path)
        for i in tqdm(range(0, fas.shape[1], self.__lambda_dn), desc='%s->angular_spectra' % self._current_res_dir):
            spectrum = fas[:, i]

            plt.figure(figsize=(20, 10))
            plt.plot(angles, spectrum, color='black', linewidth=5, linestyle='solid')

            plt.ylim([min_val - delta, max_val + delta])

            plt.xticks(fontsize=20, fontweight='bold')
            plt.yticks(fontsize=20, fontweight='bold')

            plt.xlabel('$\mathbf{\\theta}$, rad', fontsize=30, fontweight='bold')
            plt.ylabel(ylabel, fontsize=30, fontweight='bold')

            plt.grid(linewidth=2, linestyle='dotted', color='gray', alpha=0.5)

            plt.savefig(make_path(angular_spectra_path, 'lambda=%.1fnm.png' % (lambdas[0] + i * (lambdas[1] - lambdas[0]))))
            plt.close()

    def _process(self):
        files = self._get_files()
        if not files:
            raise Exception('No files detected!')

        # get data
        data = self.__transform_data(self.__get_data(files))

        # lambdas
        lambdas = self._get_lambdas(files[0])

        # steps and angles
        steps = array(list(data.keys()), dtype=float64)
        angles = self.__steps2angles(steps)

        # spectrum
        fas = array(list(data.values()), dtype=float64)

        # check
        self.__check(steps, lambdas, fas)

        # make spectrum uniform along both axes
        angles, fas = self.__make_uniform_along_angle(angles, fas)
        lambdas, fas = self.__make_uniform_along_lambda(lambdas, fas)

        # cut steps overlap
        angles, fas = self.__cut_steps_overlap(angles, fas)

        # gaussian smooth
        dangle = angles[1] - angles[0]
        dlambda = lambdas[1] - lambdas[0]
        fas = self.__smooth_spectrum(dangle, dlambda, fas)

        # reflect
        angles, fas = self.__reflect(angles, fas)

        # logarithm spectrum
        fas = self._logarithm(fas) if self._log_scale else self._normalize(fas)

        # plot
        self.__plot(angles, lambdas, fas)
