"""
Backprop NN training on bank data
"""
import os
import csv
import time
import sys
sys.path.append('./ABAGAIL/ABAGAIL.jar')
from ann import train, initialize_instances, errorOnDataSet
from func.nn.backprop import BackPropagationNetworkFactory
from shared import SumOfSquaresError, DataSet
from opt.example import NeuralNetworkOptimizationProblem
from func.nn.backprop import RPROPUpdateRule, BatchBackPropagationTrainer
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
from func.nn.activation import HyperbolicTangentSigmoid

# Network parameters found "optimal" in Assignment 1
INPUT_LAYER = 61
HIDDEN_LAYER1 = 2
HIDDEN_LAYER2 = 2
HIDDEN_LAYER3 = 2
HIDDEN_LAYER4 = 2
OUTPUT_LAYER = 1
TRAINING_ITERATIONS = 5000
OUTFILE = './../logs/XXX_LOG.csv'

def main(P,mate,mutate):
    """Run this experiment"""
    training_ints = initialize_instances('./../data/x_train_val.csv')
    testing_ints = initialize_instances('./../data/x_test.csv')
    factory = BackPropagationNetworkFactory()
    measure = SumOfSquaresError()
    data_set = DataSet(training_ints)
    acti = HyperbolicTangentSigmoid()
    rule = RPROPUpdateRule()
    oa_name = "GA_{}_{}_{}".format(P,mate,mutate)
    FILE = OUTFILE.replace('XXX',oa_name)
    with open(FILE,'w') as f:
        f.write('{},{},{},{},{},{},{},{},{}\n'.format('iteration','MSE_trg','MSE_tst','acc_trg','acc_tst','f1_trg', 'f1_tst', 'train_time','pred_time'))
    classification_network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER1,HIDDEN_LAYER2,HIDDEN_LAYER3,HIDDEN_LAYER4, OUTPUT_LAYER],acti)
    nnop = NeuralNetworkOptimizationProblem(data_set, classification_network, measure)
    oa = StandardGeneticAlgorithm(P, mate, mutate, nnop)
    train(oa, classification_network, oa_name, training_ints, testing_ints, measure, TRAINING_ITERATIONS, FILE)



if __name__ == "__main__":
    # for p in [100]:
    #     for mate in [20]:
    #         for mutate in [2,5,8,12,15]:
    for p in [50]:
        for mate in [5]:
            for mutate in [1,3,4,5]:

                args = (p,mate,mutate)
                main(*args)