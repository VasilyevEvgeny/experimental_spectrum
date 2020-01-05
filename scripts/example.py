from core import ProcessorFAS, ProcessorIFS

gauss_fas = ProcessorFAS('scripts/example_data/gauss_fas', log_scale=True)
gauss_fs = ProcessorIFS('scripts/example_data/gauss_ifs', log_scale=True)

vortex_fas = ProcessorFAS('scripts/example_data/vortex_fas', log_scale=True)
vortex_fs = ProcessorIFS('scripts/example_data/vortex_ifs', log_scale=True)
