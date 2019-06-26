#!/usr/bin/env python
from __future__ import division
import numpy
import scipy.io as sio
import matrix_utils as mu
import timeit 
from keras.layers import Input, Dense
from keras.models import Model
import keras.utils.np_utils as np_utils
import logging 
from call_back_class import RecordBestAccuracy

def init_population(coefficient_number, coefficient_bit_length, selected_basis_number, population_size):
    population = numpy.random.uniform(0.0, 1.0, [population_size, coefficient_number*coefficient_bit_length+selected_basis_number])
    m,n = population.shape
    for i in range(m):
        for j in range(n):
            if population[i][j] > 0.5:
                population[i][j] = 1
            else:
                population[i][j] = 0
    return numpy.asarray(population, numpy.int8)

def evaluate(population, space, coefficient_number, coefficient_bit_length, selected_basis_number, train_data, train_label, test_data, test_label):
    scale_value = mu.bin2int(numpy.ones(coefficient_bit_length-1))
    fitness = numpy.zeros(population.shape[0])
    mse = numpy.zeros(population.shape[0])
    weights = []
    for k in range(population.shape[0]):
        print 'begin evaluate %d/%d'%(k+1, population.shape[0])
        pop = population[k]
        coefficients = numpy.zeros(coefficient_number)
        for i in range(coefficient_number):
            coefficients[i] = mu.bin2int(pop[i*coefficient_bit_length:(i+1)*coefficient_bit_length])
        coefficients = coefficients.dot(1.0/scale_value)    
        v = space.dot(coefficients)
        null_v = mu.null(numpy.asarray([v])).T
        weight = numpy.row_stack((v, null_v)).T # each column is a basis
        selected_index = []
        for i in range(coefficient_number*coefficient_bit_length, coefficient_number*coefficient_bit_length+selected_basis_number):
            if pop[i] == int(1):
                selected_index.append(i-coefficient_number*coefficient_bit_length)
        used_weight = weight[:, selected_index]
        z = train_data.dot(used_weight)
        a = mu.relu(z)
        err = numpy.mean(numpy.sum(((train_data - a.dot(used_weight.T))**2), 1))
        # calculate the classification rate
        accuracy = mu.svm_predict(train_data, train_label, test_data, test_label, used_weight, False)
        fitness[k] = accuracy
        mse[k] = err
        weights.append(used_weight)
        #here add the training for autoencoder based on keras
    return fitness, mse, weights

def genetic_operator(population, fitness, crossover_rate, mutation_rate, coefficient_number, coefficient_bit_length, selected_basis_number, space, train_data, train_label, test_data, test_label):
    offsprings = numpy.zeros_like(population, numpy.int8)
    for i in range(offsprings.shape[0]-1):
        two_id1 = numpy.random.permutation(range(population.shape[0]))
        two_id2 = numpy.random.permutation(range(population.shape[0]))
        selected_id1 = -1
        selected_id2 = -1
        if fitness[two_id1[0]] > fitness[two_id1[1]]:
            selected_id1 = two_id1[0]
        else:
            selected_id1 = two_id1[1]
        if fitness[two_id2[0]] > fitness[two_id2[1]]:
            selected_id2 = two_id2[0]
        else:
            selected_id2 = two_id2[1]
        off1 = population[selected_id1,:]
        off2 = population[selected_id2,:]            
        # crossover
        if numpy.random.rand() < crossover_rate:
            change_point1 = numpy.random.randint(0, coefficient_number*coefficient_bit_length)
            change_point2 = numpy.random.randint(coefficient_number*coefficient_bit_length, coefficient_number*coefficient_bit_length+coefficient_bit_length)
            off1[change_point1:change_point2] = population[selected_id2,change_point1:change_point2]
            off2[change_point1:change_point2] = population[selected_id1,change_point1:change_point2]
            selected_id_index = numpy.random.randint(0, 2, size=1)[0]
            if selected_id_index == 0:
                offsprings[i, :] = off1
            else:
                offsprings[i, :] = off2
            
#             new_offspring = numpy.row_stack((off1, off2))
#             new_fitness,_,_ = evaluate(new_offspring, space, coefficient_number, coefficient_bit_length, selected_basis_number, train_data, train_label, test_data, test_label)
#             if new_fitness[0] > new_fitness[1]:
#                 offsprings[i, :] = off1
#             else:
#                 offsprings[i, :] = off2
        else:
            if fitness[selected_id1] > fitness[selected_id2]:
                offsprings[i, :] = off1
            else:
                offsprings[i, :] = off2
                
        # mutation
        rand_rs = numpy.random.rand(coefficient_number*coefficient_bit_length+coefficient_bit_length)
        for j in range(coefficient_number*coefficient_bit_length+coefficient_bit_length):
            if rand_rs[j] < mutation_rate:
                if offsprings[i, j] == 1:
                    offsprings[i, j] = 0
                else:
                    offsprings[i, j] = 1
    
    offsprings[i+1,:] = population[numpy.argmax(fitness),:]   
    return offsprings 

def evolve(train_data, train_label, test_data, test_label, population_size, generations, crossover_rate, mutation_rate):
    #history
    best_fitness = []
    best_mse = []
    #print train_data.shape, train_label.shape, test_data.shape, test_label.shape
    data_dimension = train_data.shape[1]
    space = numpy.eye(data_dimension)
    coefficient_number = data_dimension
    coefficient_bit_length = 11
    selected_basis_number = coefficient_number - 1
    print 'Initialize the population...'
    population = init_population(coefficient_number, coefficient_bit_length, selected_basis_number, population_size)
    fitness, mse, weights = evaluate(population, space, coefficient_number, coefficient_bit_length, selected_basis_number, train_data, train_label, test_data, test_label)
    best_index = numpy.argmax(fitness)
    best_weight = weights[best_index]
    best_fitness.append(fitness[best_index])
    best_mse.append(mse[best_index])
    for gen in range(generations):
        # crossover and mutation
        print 'begin current generation:%d/%d'%(gen+1,generations)
        offsprings = genetic_operator(population, fitness, crossover_rate, mutation_rate,coefficient_number, coefficient_bit_length, selected_basis_number, space, train_data, train_label, test_data, test_label)
        population = offsprings
        fitness, mse, weights =  evaluate(population, space, coefficient_number, coefficient_bit_length, selected_basis_number, train_data, train_label, test_data, test_label)
        best_index = numpy.argmax(fitness)
        best_weight = weights[best_index]
        best_fitness.append(fitness[best_index])
        best_mse.append(mse[best_index])
        
        
    return best_fitness, best_weight, best_mse    

def train_autoencoder_one_layer_svm(train_data, train_label, test_data, test_label, svm_train_data, svm_train_label, svm_test_data, svm_test_label, best_weight, train_ae_epoch):
    input_dim = train_data.shape[1]
    hidden_dim = best_weight.shape[1]
    inputs = Input(shape=(input_dim,))
    encoder = Dense(hidden_dim, activation="relu")(inputs)
    decoder = Dense(input_dim, activation="sigmoid")(encoder)
    model = Model(input=inputs, output=decoder)
    # replace the weight
    total_weights = model.get_weights()
    total_weights[0] = best_weight
    model.set_weights(total_weights)
    model.compile(optimizer="Adagrad", loss="mse")
    model.fit(train_data, train_data, batch_size=256, nb_epoch=train_ae_epoch, verbose=0)
    accuracy = mu.svm_predict(svm_train_data, svm_train_label, svm_test_data, svm_test_label, model.get_weights()[0], False)
    best_weight = model.get_weights()[0]
    return accuracy, best_weight

def train_autoencoder_one_layer_softmax(train_data, train_label, test_data, test_label, best_weight, num_class, train_ae_one_layer_softmax_epoch):
    input_dim = train_data.shape[1]
    hidden_dim = best_weight.shape[1]
    inputs = Input(shape=(input_dim,))
    encoder = Dense(hidden_dim, activation="relu")(inputs)
    predict_layer = Dense(num_class, activation="softmax")(encoder)
    model = Model(input=inputs, output=predict_layer)
    total_weights = model.get_weights()
    total_weights[0] = best_weight
    model.set_weights(total_weights)
    model.compile(optimizer="rmsprop", loss='categorical_crossentropy', metrics=['accuracy'])
    new_test_label = np_utils.to_categorical(test_label, num_class)
    new_train_label = np_utils.to_categorical(train_label, num_class)
    record_callback = RecordBestAccuracy()
    model.fit(train_data, new_train_label, nb_epoch=train_ae_one_layer_softmax_epoch, batch_size=256, validation_data=[test_data, new_test_label], callbacks=[record_callback], verbose=0)
    accuracy = model.evaluate(test_data, new_test_label, batch_size=256, verbose=0)
    return accuracy[1], record_callback.best_accuracy, model.get_weights()[0]


         
def run_one_layer(train_data, train_label, test_data, test_label, svm_train_data, svm_train_label, svm_test_data, svm_test_label, population_size, generations, num_class, train_ae_one_layer_epoch, train_ae_one_layer_softmax_epoch):
    logger = logging.getLogger('layer1-logger')  
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('layer1-logger.txt')  
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()  
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  
    fh.setFormatter(formatter)  
    ch.setFormatter(formatter)
    logger.addHandler(fh)  
    logger.addHandler(ch)      
    start_time = timeit.default_timer()
    # for GA
    logger.info("Begin for GA to find a structure and initialized weight")
    best_fitness, best_weight, best_mse = evolve(svm_train_data, svm_train_label, svm_test_data, svm_test_label, population_size, generations, crossover_rate=0.8, mutation_rate=0.01)
    numpy.savetxt('GA_best_fitness.txt', numpy.asarray(best_fitness))
    numpy.savetxt('GA_best_weight.txt', numpy.asarray(best_weight))
    numpy.savetxt('GA_best_mse.txt', numpy.asarray(best_mse))
    end_time = timeit.default_timer()
    print 'GA:total seconds:%f'%(end_time-start_time)
    logger.info('End for GA, time %f'%(end_time-start_time))
    # for train AutoEncoder+SVM
    logger.info("Begin for one layer auto-encoder and linear svm")
    start_time = timeit.default_timer()
    best_ae_one_layer_svm_accuracy, best_ae_one_layer_weight = train_autoencoder_one_layer_svm(train_data, train_label, test_data, test_label, svm_train_data, svm_train_label, svm_test_data, svm_test_label, best_weight, train_ae_one_layer_epoch)
    numpy.savetxt('1_AE_SVM_best_acc.txt', numpy.asarray([best_ae_one_layer_svm_accuracy]))
    numpy.savetxt('1_AE_SVM_best_weight.txt', numpy.asarray(best_ae_one_layer_weight))
    end_time = timeit.default_timer()
    print '1_AE_SVM:total seconds:%f'%(end_time-start_time)
    logger.info('End for one layer auto-encoder and linear svm, time %f'%(end_time-start_time))
    # for train AutoEncoder+softmax
    logger.info("Begin for one layer auto-encoder and softmax")
    start_time = timeit.default_timer()
    best_ae_one_layer_softmax_accuracy,best_ae_one_layer_softmax_history_best,softmax_best_weight = train_autoencoder_one_layer_softmax(train_data, train_label, test_data, test_label, best_weight, num_class, train_ae_one_layer_softmax_epoch)
    numpy.savetxt('1_AE_SOFTMAX_best_acc.txt', numpy.asarray([best_ae_one_layer_softmax_accuracy, best_ae_one_layer_softmax_history_best]))
    numpy.savetxt('1_AE_SOFTMAX_best_weight.txt', numpy.asarray(softmax_best_weight))
    end_time = timeit.default_timer()
    print '1_AE_SOFTMAX:total seconds:%f'%(end_time-start_time)
    logger.info('End for one layer auto-encoder and softmax, time %f'%(end_time-start_time)) 
    

if __name__ =='__main__':
    population_size = 20
    generations = 50
    
    train_ae_one_layer_epoch = 20
    train_ae_one_layer_softmax_epoch = 20
    train_ae_two_layer_epoch = 20
    train_ae_two_layer_softmax_epoch = 20
    train_ae_three_layer_epoch = 20
    train_ae_three_layer_softmax_epoch = 20
    data = sio.loadmat('./data/mnist_basic.mat')
    train_data = data['train_data']
    train_label = data['train_label'][:,0]
    test_data = data['test_data']
    test_label = data['test_label'][:,0]
    svm_train_data = data['svm_train_data']
    svm_train_label = data['svm_train_label'][:,0]
    svm_test_data = data['svm_test_data']
    svm_test_label = data['svm_test_label'][:,0]
    num_class = numpy.unique(train_label).shape[0]
    run_one_layer(train_data, train_label, test_data, test_label, svm_train_data, svm_train_label, svm_test_data, svm_test_label, population_size, generations, num_class, train_ae_one_layer_epoch, train_ae_one_layer_softmax_epoch)

    
