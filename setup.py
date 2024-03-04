from setuptools import setup, find_packages
def get_requirements(filename):
    requts = []
    with open(filename) as f:
        for line in f:
            if not line.startswith('#') and line.strip() and line.strip() !='- e .':
                requts.append(line.strip())
    return requts

setup(
    name='mlproject_E2E',
    version='0.1',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)
