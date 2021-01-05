from core import ProcessorFAS, ProcessorIFS, plot_ifs_comparison

# #
# # experiment 2019
# #
#
# gauss_fas = ProcessorFAS('scripts/experiment_2019/gauss_fas')
# gauss_fs = ProcessorIFS('scripts/experiment_2019/gauss_ifs')
#
# vortex_fas = ProcessorFAS('scripts/experiment_2019/vortex_fas')
# vortex_fs = ProcessorIFS('scripts/experiment_2019/vortex_ifs')



#
# experiment 2020
#

fas_high_energy = ProcessorFAS('scripts/experiment_2020/fas/high_energy',
                                direction='backward',
                                steps_overlap=0,
                                max_angle=0.02,
                                fix_plot_param=2)

fas_mid_energy = ProcessorFAS('scripts/experiment_2020/fas/mid_energy',
                               direction='backward',
                               steps_overlap=0,
                               max_angle=0.02,
                               fix_plot_param=1)

fas_low_energy = ProcessorFAS('scripts/experiment_2020/fas/low_energy',
                               direction='backward',
                               steps_overlap=0,
                               max_angle=0.02,
                               fix_plot_param=1)

# ifs = ProcessorIFS('scripts/experiment_2020/ifs')


#
# ifs comparison
#

plot_ifs_comparison([fas_high_energy, fas_mid_energy, fas_low_energy],
                    ['red', 'green', 'blue'],
                    ['E=8 мкДж', 'E=6 мкДж', 'E=3.75 мкДж'],
                    'scripts/experiment_2020')

#
#
# plot_ifs_comparison([ifs],
#                     ['red', 'green', 'blue', 'magenta'],
#                     ['E=8 мкДж', 'E=6 мкДж', 'E=4.4 мкДж', 'E=4 мкДж'],
#                     'scripts/experiment_2020')
