import torch

## Hook function
class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []

class SaveInput:
    def __init__(self):
        self.inputs = []

    def __call__(self, module, module_in, module_out):
        self.inputs.append(module_in)

    def clear(self):
        self.inputs = []