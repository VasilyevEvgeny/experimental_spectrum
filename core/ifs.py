from tqdm import tqdm
from os.path import join as make_path
from scipy.ndimage import gaussian_filter
import numpy as np
from matplotlib import pyplot as plt

from .base import BaseProcessor


class ProcessorIFS(BaseProcessor):
    """Integrated frequency spectrum"""

    def __init__(self, experimental_data_dir, **kwargs):
        super().__init__(experimental_data_dir, **kwargs)

        self._process()

    def __smooth_spectrum(self, dlambda, spectrum):
        n_sigma_lambda = self._sigma_lambda / dlambda

        return gaussian_filter(spectrum, sigma=(n_sigma_lambda,))

    def __plot(self, filename, lambdas, spectrum):

        ylabel = 'lg(S/S$\mathbf{_{max}}$)' if self._log_scale else 'S/S$\mathbf{_{max}}$'
        min_val, max_val = np.min(spectrum), np.max(spectrum)
        delta = 0.1 * (max_val - min_val)

        fig = plt.figure(figsize=(20, 10))
        plt.plot(lambdas, spectrum, color='black', linewidth=7, linestyle='solid')

        plt.ylim([min_val - delta, max_val + delta])

        plt.xticks(fontsize=40)
        plt.yticks(fontsize=40)

        plt.xlabel('$\mathbf{\lambda}$, nm', fontsize=55, fontweight='bold')
        plt.ylabel(ylabel, fontsize=55, fontweight='bold')

        plt.grid(linewidth=4, linestyle='dotted', color='gray', alpha=0.5)

        bbox = fig.bbox_inches.from_bounds(0, -0.4, 19, 10)

        plt.savefig(make_path(self._current_res_dir, filename), bbox_inches=bbox)
        plt.close()

    def _process(self):
        files = self._get_files()
        if not files:
            raise Exception('No files detected!')

        # lambdas
        lambdas = self._get_lambdas(files[0])

        for file in tqdm(files, desc='%s->integrated_frequency_spectrum' % self._current_res_dir):
            filename = (file.split('/')[-1]).split('.')[0]

            # spectrum
            spectrum = self._get_spectrum(file)

            # smoothing
            dlambda = lambdas[1] - lambdas[0]
            spectrum = self.__smooth_spectrum(dlambda, spectrum)

            # logarithm
            spectrum = self._logarithm(spectrum) if self._log_scale else self._normalize(spectrum)

            # plot
            self.__plot(filename, lambdas, spectrum)
