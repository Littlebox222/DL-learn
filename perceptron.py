#!/usr/bin/python
# -*- coding: UTF-8 -*-
 
class Perceptron(object):

    def __init__(self, input_num, activator):
        
        self.activator = activator
        self.weights = [0.0 for _ in range(input_num)]
        self.bias = 0.0

    def __str__(self):
        return "weights:\t:%s\nbias:\t\t:%f\n" % (self.weights, self.bias)

    def predict(self, input_vec):
        vec_multi = map(lambda (x, w): x * w, zip(input_vec, self.weights))
        signal = reduce(lambda a, b: a + b, vec_multi) + self.bias
        return self.activator(signal)

    def train(self, input_vecs, labels, iteration, rate):
        for i in range(iteration):
            self.__one_iteration(input_vecs, labels, rate)

    def __one_iteration(self, input_vecs, labels, rate):
        samples = zip(input_vecs, labels)
        for (input_vec, label) in samples:
            output = self.predict(input_vec)
            self._update_weights(input_vec, output, label, rate)

    def _update_weights(self, input_vec, output, label, rate):
        delta = label - output
        self.weights = map(lambda (x, w) : w + rate * delta * x, zip(input_vec, self.weights))
        self.bias += rate * delta
        
def f(x):
    return 1 if x > 0 else 0

def get_training_dataset():
    input_vecs = [[0,0,0], [0,0,1], [0,1,0], [1,0,0], [0,1,1], [1,0,1], [1,1,0], [1,1,1]]
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    return input_vecs, labels

def train_and_preceptron():
    p = Perceptron(3, f)
    input_vecs, labels = get_training_dataset()
    p.train(input_vecs, labels, 10, 0.1)
    return p

if __name__ == '__main__':
    and_perceptron = train_and_preceptron()

    print and_perceptron

    print "1 && 1 = %d" % and_perceptron.predict([1,1,1])
    print "1 && 0 = %d" % and_perceptron.predict([1,0,1])
    print "0 && 1 = %d" % and_perceptron.predict([0,1,0])
    print "0 && 0 = %d" % and_perceptron.predict([1,0,0])