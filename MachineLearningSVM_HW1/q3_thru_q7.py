import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

from sklearn import svm
from scipy import io



############################### -- MNIST QUESTION 3A SECTION START -- ###################################
#loading data from npz file and reshaping it
mnist_file = np.load('../data/mnist-data.npz')
reshaped_mnist_file = np.zeros((60000, 784))

#reshaping each data point within 'training_data' from 28x28 to 784x1
counter = 0
for sample in mnist_file['training_data']:
    reshaped_mnist_file[counter] = np.reshape(sample, (784))
    counter += 1

#defining deterministic random seed 
np.random.seed(55)

#getting all the indices of the training data, then shuffling the indices
mnist_idxs = np.arange(len(mnist_file['training_data']))
np.random.shuffle(mnist_idxs)
shuffled_mnist_data = reshaped_mnist_file[mnist_idxs]
shuffled_mnist_labels = mnist_file['training_labels'][mnist_idxs]

#partitioning mnist data
mnist_idx_cutoff = 10000
mnist_val_labels = shuffled_mnist_labels[:mnist_idx_cutoff]
mnist_val_data = shuffled_mnist_data[:mnist_idx_cutoff]
mnist_tr_labels = shuffled_mnist_labels[mnist_idx_cutoff:]
mnist_tr_data = shuffled_mnist_data[mnist_idx_cutoff:]
############################### -- MNIST QUESTION 3A SECTION END -- ###################################

############################### -- SPAM QUESTION 3A SECTION START -- ###################################

#defining deterministic random seed 
np.random.seed(66)

#getting all the indices of the training data, then shuffling the indices
spam_file = np.load('../data/spam-data.npz')
spam_idxs = np.arange(len(spam_file['training_data']))
np.random.shuffle(spam_idxs)
spam_data = spam_file['training_data'][spam_idxs]
spam_labels = spam_file['training_labels'][spam_idxs]

#partitioning spam data
spam_idx_cutoff = int(0.2 * len(spam_idxs))
spam_val_labels = spam_labels[:spam_idx_cutoff]
spam_val_data = spam_data[:spam_idx_cutoff]
spam_tr_labels = spam_labels[spam_idx_cutoff:]
spam_tr_data = spam_data[spam_idx_cutoff:]
############################### -- SPAM QUESTION 3A SECTION END -- ###################################

############################### -- QUESTION 3B EVAL MODEL SECTION START -- ###################################
def eval_metric(val_labels, predicted_labels):
    total = 0
    for i in range(len(val_labels)):
        if val_labels[i] == predicted_labels[i]:
            total += 1
    return (total / len(val_labels))
############################### -- QUESTION 3B EVAL MODEL SECTION END -- ###################################

############# -- QUESTION 4 PLOTTING TRAINING/VALIDATION ACCURACY SECTION START -- ###############
#train and plot mnist data given a sample number
def mnist_train_model_AND_plot(sample_size):

    mnist_model = svm.LinearSVC()
    mnist_tr_data_sample = mnist_tr_data[:sample_size]
    mnist_tr_labels_sample = mnist_tr_labels[:sample_size]
    mnist_model.fit(mnist_tr_data_sample, mnist_tr_labels_sample)
    predicted_val_labels = mnist_model.predict(mnist_val_data)
    predicted_tr_labels = mnist_model.predict(mnist_tr_data_sample)
    val_data_acc_rate = eval_metric(mnist_val_labels, predicted_val_labels)
    tr_data_acc_rate = eval_metric(mnist_tr_labels_sample, predicted_tr_labels)
    plt.plot(sample_size, val_data_acc_rate, 'go')
    plt.plot(sample_size, tr_data_acc_rate, 'yo')

# !!!!!!!!!!!!!!!! COMMENT AND UNCOMMENT THIS SECTION OF CODE TO TEST QUESTION 4 !!!!!!!!!!!!!!!! #

# mnist_train_model_AND_plot(100)
# mnist_train_model_AND_plot(200)
# mnist_train_model_AND_plot(500)
# mnist_train_model_AND_plot(1000)
# mnist_train_model_AND_plot(2000)
# mnist_train_model_AND_plot(5000)
# mnist_train_model_AND_plot(10000)
# plt.title('MNIST Model Accuracy')
# plt.show() #this show() will display mnist data points

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#train and plot spam data given a sample number
def spam_train_model_AND_plot(sample_size):
    
    spam_model = svm.LinearSVC()
    spam_tr_data_sample = spam_tr_data[:sample_size] #data dims: 100,1,28,28
    spam_tr_labels_sample = spam_tr_labels[:sample_size]

    print("data:", np.shape(spam_tr_data_sample))
    print("labels:", np.shape(spam_tr_labels_sample))

    #make model fit the data onto the labels
    spam_model.fit(spam_tr_data_sample, spam_tr_labels_sample)

    #predict training labels and evaluate training accuracy
    predicted_tr_labels = spam_model.predict(spam_tr_data_sample)
    tr_data_acc_rate = eval_metric(spam_tr_labels_sample, predicted_tr_labels)

    #predict validation labels and evaluate validation accuracy
    predicted_val_labels = spam_model.predict(spam_val_data)
    val_data_acc_rate = eval_metric(spam_val_labels, predicted_val_labels)

    plt.plot(sample_size, val_data_acc_rate, 'go')
    plt.plot(sample_size, tr_data_acc_rate, 'yo')

# !!!!!!!!!!!!!!!! COMMENT AND UNCOMMENT THIS SECTION OF CODE TO TEST QUESTION 4 !!!!!!!!!!!!!!!! #

# spam_train_model_AND_plot(100)
# spam_train_model_AND_plot(200)
# spam_train_model_AND_plot(500)
# spam_train_model_AND_plot(1000)
# spam_train_model_AND_plot(2000)
# spam_train_model_AND_plot(len(spam_tr_data))
# plt.title('SPAM Model Accuracy')
# plt.show() #this show() will display spam data points
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1

############# -- QUESTION 4 PLOTTING TRAINING/VALIDATION ACCURACY SECTION END -- ###############

############# -- QUESTION 5 MNIST HYPERPARAMETER TUNING SECTION START -- ###############

#sweeps through given range of c values, with
#args:
    #start - start c value
    #end - end c value
    #step - step increase size between c values
    #N - number of best c_values you want to return
#this func will take in these arguments and return a list of the best c_values (NOT IN ORDER OF BEST TO WORST, THOUGH. JUST RANDOM) 
def mnist_c_value_sweeping(start, end, step, N, sample_size):
    #then, to find the N best c_values, you find the max() of accuracy_list and use index() to find the index, which
    #then helps you to calculate the c_value associated with the accuracy, which you will store in a list
    #then, after finding the c_value, pop out the max accuracy you found and continue to iteratively repeat the process N times.
    temp_dict = dict() # {c_value: accuracy rate}
    #refined_c_value_list = [] #store top N c_values
    iteration_amt = int((end - start)/step) #ex: [-5, 5] w step size 2 => 5 iterations
    #current_c_value = start + (current_iteration*step)
    for current_iteration in range(iteration_amt):
        current_c_value = start + (current_iteration*step)
        print("curr c val:", current_c_value)
        temp_dict[current_c_value] = mnist_with_c_value(current_c_value, sample_size)
        print("output from mnist_w_c_value func", mnist_with_c_value(current_c_value, sample_size))
    
    #citation for sorting dictionary items in reverse order: https://www.freecodecamp.org/news/sort-dictionary-by-value-in-python/
    sorted_dict = dict(sorted(temp_dict.items(), key=lambda x:x[1], reverse=True))
    #final_dict = {sorted_dict}
    print("c_val to acc sorted dictionary:", sorted_dict)
    temp_list = list(sorted_dict.keys())
    refined_c_value_list = temp_list[:N]
    print(refined_c_value_list)

#trains the mnist model with the inputted c_value and returns the accuracy
def mnist_with_c_value(c_val, sample_size):

    mnist_tr_data_sample = mnist_tr_data[:sample_size]
    mnist_tr_labels_sample = mnist_tr_labels[:sample_size]
    mnist_model = svm.LinearSVC(C=c_val)
    mnist_model.fit(mnist_tr_data_sample, mnist_tr_labels_sample)
    predict_val_labels = mnist_model.predict(mnist_val_data)
    print(eval_metric(mnist_val_labels, predict_val_labels))
    return eval_metric(mnist_val_labels, predict_val_labels)


# !!!!!!!!!!!!!!!! COMMENT AND UNCOMMENT THIS SECTION OF CODE TO TEST QUESTION 5 !!!!!!!!!!!!!!!! #
#mnist_c_value_sweeping(0.000001, 1, .1, 10, 10000)
#mnist_with_c_value(0.000001, 30000)
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

############# -- QUESTION 5 MNIST HYPERPARAMETER TUNING SECTION END -- ###############


############# -- QUESTION 6 K-FOLD CROSS VALIDATION SECTION START -- ###############

def spam_k_fold_cross_val(c_val, k):
    model_acc = 0
    #split the training data into k partitions
    k_partitions = np.array_split(range(len(spam_data)), k)

    for i in range(k):
        #selecting i to be the validation set, and the rest of the partitions to be the training data
        val_indices = k_partitions[i]
        val_data = spam_data[val_indices]
        val_labels = spam_labels[val_indices]
        
        #putting all training set indices into a list, then condensing the training data into a single list
        tr_idxs = np.concatenate([k_partitions[j] for j in range(5) if j != i])
        tr_data = spam_data[tr_idxs]
        tr_labels = spam_labels[tr_idxs]

        #training model with the c value
        spam_model = svm.LinearSVC(C=c_val)
        spam_model.fit(tr_data, tr_labels)
        predict_val_labels = spam_model.predict(val_data)
        model_acc += eval_metric(predict_val_labels, val_labels)
    print("cval:", c_val, "acc:", model_acc/5)


# !!!!!!!!!!!!!!!! COMMENT AND UNCOMMENT THIS SECTION OF CODE TO TEST QUESTION 6 !!!!!!!!!!!!!!!! #
#spam_k_fold_cross_val(87, 5)
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

############# -- QUESTION 6 K-FOLD CROSS VALIDATION SECTION END -- ###############
    
############# -- QUESTION 7 KAGGLE SUBMISSION WITH BEST MODEL(S) SECTION START -- ###############
def mnist_results_to_csv(y_test):
    y_test = y_test.astype(int)
    df = pd.DataFrame({'Category': y_test})
    df.index += 1 # Ensures that the index starts at 1
    df.to_csv('mnist_submission1.csv', index_label='Id')

def spam_results_to_csv(y_test):
    y_test = y_test.astype(int)
    df = pd.DataFrame({'Category': y_test})
    df.index += 1 # Ensures that the index starts at 1
    df.to_csv('spam_submission1.csv', index_label='Id')    


def spam_with_c_value(c_val, sample_size):

    spam_tr_data_sample = spam_tr_data[:sample_size]
    spam_tr_labels_sample = spam_tr_labels[:sample_size]
    spam_model = svm.LinearSVC(C=c_val)
    spam_model.fit(spam_tr_data_sample, spam_tr_labels_sample)
    predict_val_labels = spam_model.predict(spam_val_data)
    print(eval_metric(spam_val_labels, predict_val_labels))
    return eval_metric(spam_val_labels, predict_val_labels)

def spam_c_value_sweeping(start, end, step, N, sample_size):
    #then, to find the N best c_values, you find the max() of accuracy_list and use index() to find the index, which
    #then helps you to calculate the c_value associated with the accuracy, which you will store in a list
    #then, after finding the c_value, pop out the max accuracy you found and continue to iteratively repeat the process N times.
    temp_dict = dict() # {c_value: accuracy rate}
    #refined_c_value_list = [] #store top N c_values
    iteration_amt = int((end - start)/step) #ex: [-5, 5] w step size 2 => 5 iterations
    #current_c_value = start + (current_iteration*step)
    for current_iteration in range(iteration_amt):
        current_c_value = start + (current_iteration*step)
        print("curr c val:", current_c_value)
        temp_dict[current_c_value] = spam_with_c_value(current_c_value, sample_size)
        print("output from spam_w_c_value func", spam_with_c_value(current_c_value, sample_size))
    
    #citation for sorting dictionary items in reverse order: https://www.freecodecamp.org/news/sort-dictionary-by-value-in-python/
    sorted_dict = dict(sorted(temp_dict.items(), key=lambda x:x[1], reverse=True))
    #final_dict = {sorted_dict}
    print("c_val to acc sorted dictionary:", sorted_dict)
    temp_list = list(sorted_dict.keys())
    refined_c_value_list = temp_list[:N]
    print(refined_c_value_list)



def spam_csv_save():
    np.random.seed(66)

    #getting all the indices of the training data, then shuffling the indices
    spam_file = np.load('../data/spam-data.npz')
    spam_idxs = np.arange(len(spam_file['training_data']))
    np.random.shuffle(spam_idxs)
    spam_data = spam_file['training_data'][spam_idxs]
    spam_labels = spam_file['training_labels'][spam_idxs]

    #partitioning spam data
    spam_idx_cutoff = int(0.2 * len(spam_idxs))
    spam_val_labels = spam_labels[:spam_idx_cutoff]
    spam_val_data = spam_data[:spam_idx_cutoff]
    spam_tr_labels = spam_labels[spam_idx_cutoff:]
    spam_tr_data = spam_data[spam_idx_cutoff:]

    final_spam_model = svm.LinearSVC(C=87)

    final_spam_model.fit(spam_data, spam_labels)
    predicted_test_data = final_spam_model.predict(spam_file['test_data'])
    spam_results_to_csv(predicted_test_data)

def mnist_csv_save():
    mnist_file = np.load('../data/mnist-data.npz')
    reshaped_mnist_tr_file = np.zeros((60000, 784))
    counter_tr = 0
    for sample in mnist_file['training_data']:
        reshaped_mnist_tr_file[counter_tr] = np.reshape(sample, (784))
        counter_tr += 1

    reshaped_mnist_test_file = np.zeros((10000, 784))
    counter_test = 0
    for sample in mnist_file['test_data']:
        reshaped_mnist_test_file[counter_test] = np.reshape(sample, (784))
        counter_test += 1
    #defining deterministic random seed 
    np.random.seed(55)

    mnist_tr_labels = mnist_file['training_labels']
    final_mnist_model = svm.LinearSVC(C=0.000001)
    final_mnist_model.fit(reshaped_mnist_tr_file, mnist_tr_labels)
    predicted_test_data = final_mnist_model.predict(reshaped_mnist_test_file)
    mnist_results_to_csv(predicted_test_data)


# !!!!!!!!!!!!!!!! COMMENT AND UNCOMMENT THIS SECTION OF CODE TO TEST QUESTION 6 !!!!!!!!!!!!!!!! #
# spam_csv_save()
# mnist_csv_save()
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
############# -- QUESTION 7 KAGGLE SUBMISSION WITH BEST MODEL(S) SECTION END -- ###############