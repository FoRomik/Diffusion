from setuptools import setup, find_packages

setup(
      name = "Diffusion",
      version = "1.0.0",
      description = "Steady-state diffusion equation solver.",
      author = "Jan Zmazek",
      url = "https://github.com/janzmazek/Diffusion",
      license = "MIT License",
      packages = find_packages(exclude=['*test']),
#     scripts = ['scripts/diffusion'],
#     data_files = [('/etc/diffusion' ,['config.yaml'])],
      install_requires = ['matplotlib', 'numpy', 'scipy', 'os', 'networkx', 'pymetis']
)
