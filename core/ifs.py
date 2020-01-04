from tqdm import tqdm
from os.path import join as make_path
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt

from .base import BaseProcessor


class ProcessorIFS(BaseProcessor):
    def __init__(self, experimental_data_dir, **kwargs):
        super().__init__(experimental_data_dir, **kwargs)

        self._process()

    def __smooth_spectrum(self, dlambda, spectrum):
        n_sigma_lambda = self._sigma_lambda / dlambda

        return gaussian_filter(spectrum, sigma=(n_sigma_lambda,))

    def __plot(self, filename, lambdas, spectrum):
        #
        # frequency spectra
        #

        plt.figure(figsize=(20, 10))
        plt.plot(lambdas, spectrum, color='black', linewidth=5, linestyle='solid')

        plt.ylim([self._log_power - 0.1, 0.1])

        plt.xticks(fontsize=20, fontweight='bold')
        plt.yticks(fontsize=20, fontweight='bold')

        plt.xlabel('$\mathbf{\lambda}$, nm', fontsize=30, fontweight='bold')
        plt.ylabel('lg(S/S$\mathbf{_{max}}$)', fontsize=30, fontweight='bold')

        plt.grid(linewidth=2, linestyle='dotted', color='gray', alpha=0.5)

        plt.savefig(make_path(self._current_res_dir, filename))
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
            spectrum = self._logarithm(spectrum)

            # plot
            self.__plot(filename, lambdas, spectrum)
