import tensorflow as tf

class DriftLinear:

    def __init__(self,a,b):
        self.a = a
        self.b = b

    def __call__(self,t,S):
        return self.a*S+self.b
