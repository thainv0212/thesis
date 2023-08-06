from sacred import Experiment
ex = Experiment('captured_func_demo2')

ex = Experiment('prefix_demo')

@ex.config
def my_config1():
    dataset = {
        'filename': 'foo.txt',
        'path': '/tmp/'
    }

@ex.capture(prefix='dataset')
def print_me(filename, path):  # direct access to entries of the dataset dict
    print("filename =", filename)
    print("path =", path)

@ex.automain
def main():
    print_me()