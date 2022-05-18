from collections import OrderedDict
from numba import jit
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import numpy as np
from numpy import append, save
from os.path import join as make_path
from os import mkdir
from numpy import array, zeros, float64, mean, where, deg2rad, log10
from matplotlib import pyplot as plt

from .base import BaseProcessor


from matplotlib import rc, cm

rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage[utf8]{inputenc}')
rc('text.latex', preamble=r'\usepackage[russian]{babel}')


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


class ProcessorFAS(BaseProcessor):
    """Frequency-angular spectrum"""

    def __init__(self, experimental_data_dir, **kwargs):
        super().__init__(experimental_data_dir, **kwargs)

        self.__micron_per_step = kwargs.get('micron_per_step', 10)  # [micron / step]
        self.__deg_per_micron = kwargs.get('deg_per_micron', 5 / 3500)  # [deg / micron]
        self.__direction = kwargs.get('direction', 'forward')  # direction of spectrum measurements 'forward' / 'backward'

        self.__sigma_angle = kwargs.get('sigma_angle', 0)  # [rad]
        self.__steps_overlap = kwargs.get('steps_overlap', 4)  # []
        self.__lambda_dn = kwargs.get('lambda_dn', 50)  # []

        self.__max_angle = kwargs.get('max_angle', None)  # [rad]
        self.__fix_plot_param = kwargs.get('fix_plot_param', 0)  # 0/1/2 or smth like this

        self._process()

    def __get_data(self, files):
        data = {}
        for file in files:
            step = int(''.join(filter(str.isdigit, (file.split('/')[-1]).split('.')[0])))
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

    def __check(self, steps, lambdas, spectrum):
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
        spectrum = spectrum[self.__steps_overlap:, :]

        return angles, spectrum

    def __get_fas(self, data):
        if self.__direction == 'forward':
            return array(list(data.values()), dtype=float64)
        elif self.__direction == 'backward':
            return array(list(data.values())[::-1], dtype=float64)
        else:
            raise Exception('Wrong direction!')

    @staticmethod
    def __reflect(angles, spectrum):
        angles = array([-e for e in list(angles)][::-1][:-1] + list(angles), dtype=float64)
        spectrum = array((list(spectrum)[::-1])[:-1] + list(spectrum), dtype=float64)

        return angles, spectrum

    def __add_zero_gap(self, spectrum, angles):
        cur_max_angle = angles[-1]
        dangle = angles[1] - angles[0]

        if self.__max_angle is not None and self.__max_angle > cur_max_angle:
            n_points_to_add = int((self.__max_angle - cur_max_angle) / dangle + 1)
            zero_arr = zeros(shape=(n_points_to_add, spectrum.shape[1]))
            spectrum = append(spectrum, zero_arr, axis=0)
            angles = append(angles, [angles[-1] + i * dangle for i in range(n_points_to_add)], axis=0)

        return angles, spectrum,

    def __make_ifs(self, fas):
        ifs = zeros(shape=(fas.shape[1],), dtype=float64)
        for i in range(fas.shape[0]):
            for j in range(fas.shape[1]):
                ifs[j] += fas[i][j]

        return ifs

    def __plot_ifs(self, lambdas, ifs):
        ylabel = 'lg(S/S$\mathbf{_{max}}$)' if self._log_scale else 'S/S$\mathbf{_{max}}$'
        min_val, max_val = -2, 0
        delta = 0.1 * (max_val - min_val)
        lambdas = np.array([e / 10 ** 3 for e in lambdas])
        ifs = np.array(self._logarithm(ifs) if self._log_scale else self._normalize(ifs))

        fig = plt.figure(figsize=(20, 10))
        plt.plot(lambdas, ifs, color='black', linewidth=7, linestyle='solid')

        plt.ylim([min_val - delta, max_val + delta])

        plt.xticks(fontsize=40)
        plt.yticks(fontsize=40)

        plt.xlabel('$\mathbf{\lambda}$, $\mathbf{\mu}$m', fontsize=55, fontweight='bold')
        plt.ylabel(ylabel, fontsize=55, fontweight='bold')

        plt.grid(linewidth=4, linestyle='dotted', color='gray', alpha=0.5)

        bbox = fig.bbox_inches.from_bounds(0, -0.4, 19, 10)

        plt.savefig(make_path(self._current_res_dir, 'ifs.png'), bbox_inches=bbox)
        plt.close()

    def __plot_fas_dissertation(self, angles, lambdas, fas):

        ylabel = '$\lg (S/S_{max})$' if self._log_scale else 'S/S$\mathbf{_{max}}$'
        min_val, max_val = np.min(fas), np.max(fas)
        delta = 0.1 * (max_val - min_val)

        #
        # fas
        #

        fig, ax = plt.subplots(figsize=cm2inch(5, 5))
        plot = plt.contourf(fas, cmap='gray', levels=100)

        x_ticks_labels = ['1.4', '1.8', '2.2']
        dlambda = lambdas[1] - lambdas[0]
        x_ticks = [int((float(e) * 10**3 - lambdas[0]) / dlambda) for e in x_ticks_labels]
        plt.xticks(x_ticks, x_ticks_labels, fontsize=12)

        # y_ticks_labels = ['$-$0.01', ' 0.00', '+0.01']
        y_ticks_labels = ['$-$0.04', '$-$0.02', ' 0.00', '+0.02', '+0.04']
        dangle = angles[1] - angles[0]
        max_angle = angles[-1]
        y_ticks = [int((float(e.replace('$-$', '-')) + max_angle) / dangle) + self.__fix_plot_param for e in y_ticks_labels]
        plt.yticks(y_ticks, y_ticks_labels, fontsize=12)

        plt.xlabel('$\lambda$, мкм', fontsize=14)
        plt.ylabel('$\\theta$, рад', fontsize=14)

        ax.tick_params(direction='in', colors='white', labelcolor='black', top=True, right=True)
        for spine in ax.spines.values():
            spine.set_edgecolor('white')

        n_ticks_colorbar_levels = 4
        dcb = (max_val - min_val) / n_ticks_colorbar_levels
        levels_ticks_colorbar = [min_val + i * dcb for i in range(n_ticks_colorbar_levels + 1)]

        # colorbar = fig.colorbar(plot, ticks=levels_ticks_colorbar, orientation='vertical', aspect=10, pad=0.05)
        # colorbar.set_label(ylabel, labelpad=-20, y=1.2, rotation=0, fontsize=12, fontweight='bold')
        # ticks_cbar = ['$-$%03.1f' % abs(e) if e < 0 else '0.0' for e in levels_ticks_colorbar]
        # colorbar.ax.set_yticklabels(ticks_cbar)
        # colorbar.ax.tick_params(labelsize=10)

        plt.savefig(make_path(self._current_res_dir, 'fas'), bbox_inches='tight', dpi=500)
        plt.close()

    def __plot_fas(self, angles, lambdas, fas):

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
        plt.xticks(x_ticks, x_ticks_labels, fontsize=30)

        y_ticks_labels = ['-0.01', ' 0.00', '+0.01']
        dangle = angles[1] - angles[0]
        max_angle = angles[-1]
        y_ticks = [int((float(e) + max_angle) / dangle) + self.__fix_plot_param for e in y_ticks_labels]
        plt.yticks(y_ticks, y_ticks_labels, fontsize=30)

        plt.xlabel('$\mathbf{\lambda}$, $\mathbf{\mu}$m', fontsize=40, fontweight='bold')
        plt.ylabel('$\mathbf{\\theta}$, rad', fontsize=40, fontweight='bold')

        plt.grid(linewidth=3, linestyle='dotted', color='white', alpha=0.5)

        n_ticks_colorbar_levels = 4
        dcb = (max_val - min_val) / n_ticks_colorbar_levels
        levels_ticks_colorbar = [min_val + i * dcb for i in range(n_ticks_colorbar_levels + 1)]

        colorbar = fig.colorbar(plot, ticks=levels_ticks_colorbar, orientation='vertical', aspect=10, pad=0.05)
        colorbar.set_label(ylabel, labelpad=-100, y=1.2, rotation=0, fontsize=40, fontweight='bold')
        ticks_cbar = ['%05.2f' % e if e != 0 else '00.00' for e in levels_ticks_colorbar]
        colorbar.ax.set_yticklabels(ticks_cbar)
        colorbar.ax.tick_params(labelsize=30)

        plt.savefig(make_path(self._current_res_dir, 'fas'), bbox_inches='tight', dpi=300)
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
        fas = self.__get_fas(data)

        # check
        self.__check(steps, lambdas, fas)

        # make spectrum uniform along both axes
        angles, fas = self.__make_uniform_along_angle(angles, fas)
        lambdas, fas = self.__make_uniform_along_lambda(lambdas, fas)
        self._lambdas_list.append(lambdas)

        # cut steps overlap
        angles, fas = self.__cut_steps_overlap(angles, fas)

        # gaussian smooth
        dangle = angles[1] - angles[0]
        dlambda = lambdas[1] - lambdas[0]
        fas = self.__smooth_spectrum(dangle, dlambda, fas)

        # add zero gap
        angles, fas = self.__add_zero_gap(fas, angles)

        # make ifs
        ifs = self.__make_ifs(fas)
        self._ifs_list.append(ifs)
        self.__plot_ifs(lambdas, ifs)

        np.save('{}.npy'.format(make_path(self._current_res_dir, 'ifs')), ifs)

        # reflect
        angles, fas = self.__reflect(angles, fas)

        # logarithm spectrum
        fas = self._logarithm(fas) if self._log_scale else self._normalize(fas)

        # plot
        # self.__plot_fas(angles, lambdas, fas)
        self.__plot_fas_dissertation(angles, lambdas, fas)
