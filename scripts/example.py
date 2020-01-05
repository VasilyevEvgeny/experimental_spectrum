from core import ProcessorFAS, ProcessorIFS

gauss_fas = ProcessorFAS('scripts/example_data/gauss_fas', steps_overlap=5)
gauss_fs = ProcessorIFS('scripts/example_data/gauss_ifs')

vortex_fas = ProcessorFAS('scripts/example_data/vortex_fas')
vortex_fs = ProcessorIFS('scripts/example_data/vortex_ifs')
