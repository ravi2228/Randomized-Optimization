"""
Backprop NN training on titanic data
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

def main(T, CE):
    """Run this experiment"""
    training_ints = initialize_instances('./../data/x_train_val.csv')
    testing_ints = initialize_instances('./../data/x_test.csv')
    factory = BackPropagationNetworkFactory()
    measure = SumOfSquaresError()
    data_set = DataSet(training_ints)
    acti = HyperbolicTangentSigmoid()
    rule = RPROPUpdateRule()
    oa_name = "SA_{}_{}".format(T, CE)
    FILE = OUTFILE.replace('XXX',oa_name)
    with open(FILE,'w') as f:
        f.write('{},{},{},{},{},{},{},{},{}\n'.format('iteration','MSE_trg','MSE_tst','acc_trg','acc_tst','f1_trg', 'f1_tst', 'train_time','pred_time'))
    classification_network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER1,HIDDEN_LAYER2,HIDDEN_LAYER3,HIDDEN_LAYER4, OUTPUT_LAYER],acti)
    nnop = NeuralNetworkOptimizationProblem(data_set, classification_network, measure)
    oa = SimulatedAnnealing(T, CE, nnop)
    train(oa, classification_network, oa_name, training_ints, testing_ints, measure, TRAINING_ITERATIONS, FILE)



if __name__ == "__main__":
    # with open(OUTFILE,'w') as f:
    #     f.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format('T','CE','iteration','MSE_trg','MSE_tst','acc_trg','acc_tst','f1_trg', 'f1_tst', 'train_time','pred_time'))
    #for CE in [0.35,0.55,0.7,0.95]:
    for CE in [0.70]:
        for  T in [1e6, 1e8, 1e12, 1e15]:
        #for  T in [1e10]:
            main(T, CE)