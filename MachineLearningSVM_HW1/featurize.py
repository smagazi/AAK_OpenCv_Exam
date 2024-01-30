'''
**************** PLEASE READ ***************

Script that reads in spam and ham messages and converts each training example
into a feature vector

Code intended for UC Berkeley course CS 189/289A: Machine Learning

Requirements:
-scipy ('pip install scipy')

To add your own features, create a function that takes in the raw text and
word frequency dictionary and outputs a int or float. Then add your feature
in the function 'def generate_feature_vector'

The output of your file will be a .mat file. The data will be accessible using
the following keys:
    -'training_data'
    -'training_labels'
    -'test_data'

Please direct any bugs to kevintee@berkeley.edu
'''

from collections import defaultdict
import glob
import re
import scipy.io
import numpy as np
import pdb

NUM_TRAINING_EXAMPLES = 4172
NUM_TEST_EXAMPLES = 1000

BASE_DIR = '../data/'
SPAM_DIR = 'spam/'
HAM_DIR = 'ham/'
TEST_DIR = 'test/'

# ************* Features *************

# Features that look for certain words
def freq_pain_feature(text, freq):
    return float(freq['pain'])

def freq_private_feature(text, freq):
    return float(freq['private'])

def freq_bank_feature(text, freq):
    return float(freq['bank'])

def freq_money_feature(text, freq):
    return float(freq['money'])

def freq_drug_feature(text, freq):
    return float(freq['drug'])

def freq_spam_feature(text, freq):
    return float(freq['spam'])

def freq_prescription_feature(text, freq):
    return float(freq['prescription'])

def freq_creative_feature(text, freq):
    return float(freq['creative'])

def freq_height_feature(text, freq):
    return float(freq['height'])

def freq_featured_feature(text, freq):
    return float(freq['featured'])

def freq_differ_feature(text, freq):
    return float(freq['differ'])

def freq_width_feature(text, freq):
    return float(freq['width'])

def freq_other_feature(text, freq):
    return float(freq['other'])

def freq_energy_feature(text, freq):
    return float(freq['energy'])

def freq_business_feature(text, freq):
    return float(freq['business'])

def freq_message_feature(text, freq):
    return float(freq['message'])

def freq_volumes_feature(text, freq):
    return float(freq['volumes'])

def freq_revision_feature(text, freq):
    return float(freq['revision'])

def freq_path_feature(text, freq):
    return float(freq['path'])

def freq_meter_feature(text, freq):
    return float(freq['meter'])

def freq_memo_feature(text, freq):
    return float(freq['memo'])

def freq_planning_feature(text, freq):
    return float(freq['planning'])

def freq_pleased_feature(text, freq):
    return float(freq['pleased'])

def freq_record_feature(text, freq):
    return float(freq['record'])

def freq_out_feature(text, freq):
    return float(freq['out'])


# Features that look for certain characters
def freq_semicolon_feature(text, freq):
    return text.count(';')

def freq_dollar_feature(text, freq):
    return text.count('$')

def freq_sharp_feature(text, freq):
    return text.count('#')

def freq_exclamation_feature(text, freq):
    return text.count('!')

def freq_para_feature(text, freq):
    return text.count('(')

def freq_bracket_feature(text, freq):
    return text.count('[')

def freq_and_feature(text, freq):
    return text.count('&')

def freq_question_feature(text, freq):
    return text.count('?')

# --------- Add your own feature methods ----------
def sex_feature(text, freq):
    return int('sex' in text)

def gay_feature(text, freq):
    return int('gay' in text)

def satan_feature(text, freq):
    return int('satan' in text)

def vicodin_feature(text, freq):
    return int('vicodin' in text)

def faggot_feature(text, freq):
    return int('faggot' in text)

def viagra_feature(text, freq):
    return int('viagra' in text)

def shit_feature(text, freq):
    return int('shit' in text)

def bitch_feature(text, freq):
    return int('bitch' in text)

def fuck_feature(text, freq):
    return int('fuck' in text)

def sperm_feature(text, freq):
    return int('sperm' in text)

def hook_up_feature(text, freq):
    return int('hook up' in text)

def hot_girls_feature(text, freq):
    return int('hot girls' in text)

def hookup_feature(text, freq):
    return int('hookup' in text)

def free_feature(text, freq):
    return int('free' in text)

def your_area_feature(text, freq):
    return int('your area' in text)

def money_back_feature(text, freq):
    return int('money back' in text)

def satisfied_feature(text, freq):
    return int('satisfied' in text)

def penis_feature(text, freq):
    return int('penis' in text)

def jizz_feature(text, freq):
    return int('jizz' in text)

def meds_feature(text, freq):
    return int('meds' in text)



# Generates a feature vector
def generate_feature_vector(text, freq):
    feature = []
    feature.append(freq_pain_feature(text, freq))
    feature.append(freq_private_feature(text, freq))
    feature.append(freq_bank_feature(text, freq))
    feature.append(freq_money_feature(text, freq))
    feature.append(freq_drug_feature(text, freq))
    feature.append(freq_spam_feature(text, freq))
    feature.append(freq_prescription_feature(text, freq))
    feature.append(freq_creative_feature(text, freq))
    feature.append(freq_height_feature(text, freq))
    feature.append(freq_featured_feature(text, freq))
    feature.append(freq_differ_feature(text, freq))
    feature.append(freq_width_feature(text, freq))
    #feature.append(freq_other_feature(text, freq))
    feature.append(freq_energy_feature(text, freq)) ##
    feature.append(freq_business_feature(text, freq))
    #feature.append(freq_message_feature(text, freq))
    feature.append(freq_volumes_feature(text, freq))
    feature.append(freq_revision_feature(text, freq))
    feature.append(freq_path_feature(text, freq))
    feature.append(freq_meter_feature(text, freq))
    #feature.append(freq_memo_feature(text, freq))
    #feature.append(freq_planning_feature(text, freq))
    feature.append(freq_pleased_feature(text, freq))
    #feature.append(freq_record_feature(text, freq))
    #feature.append(freq_out_feature(text, freq))
    feature.append(freq_semicolon_feature(text, freq))
    feature.append(freq_dollar_feature(text, freq))
    feature.append(freq_sharp_feature(text, freq))
    feature.append(freq_exclamation_feature(text, freq))
    feature.append(freq_para_feature(text, freq))
    feature.append(freq_bracket_feature(text, freq))
    feature.append(freq_and_feature(text, freq))
    feature.append(sex_feature(text, freq))
    #feature.append(gay_feature(text, freq))
    feature.append(satan_feature(text, freq))
    feature.append(vicodin_feature(text, freq))
    feature.append(faggot_feature(text, freq))
    feature.append(viagra_feature(text, freq))
    feature.append(shit_feature(text, freq))
    #feature.append(bitch_feature(text, freq))
    # feature.append(fuck_feature(text, freq))
    feature.append(sperm_feature(text, freq))
    feature.append(hook_up_feature(text, freq))
    feature.append(hot_girls_feature(text, freq))
    feature.append(hookup_feature(text, freq))
    feature.append(free_feature(text, freq))
    feature.append(your_area_feature(text, freq))
    feature.append(money_back_feature(text, freq))
    #feature.append(satisfied_feature(text, freq))
    feature.append(penis_feature(text, freq))
    feature.append(jizz_feature(text, freq))
    #feature.append(meds_feature(text, freq))



    # --------- Add your own features here ---------
    # Make sure type is int or float

    return feature

# This method generates a design matrix with a list of filenames
# Each file is a single training example
def generate_design_matrix(filenames):
    design_matrix = []
    for filename in filenames:
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            try:
                text = f.read() # Read in text from file
            except Exception as e:
                # skip files we have trouble reading.
                continue
            text = text.replace('\r\n', ' ') # Remove newline character
            words = re.findall(r'\w+', text)
            word_freq = defaultdict(int) # Frequency of all words
            for word in words:
                word_freq[word] += 1

            # Create a feature vector
            feature_vector = generate_feature_vector(text, word_freq)
            design_matrix.append(feature_vector)
    return design_matrix

# ************** Script starts here **************
# DO NOT MODIFY ANYTHING BELOW

spam_filenames = glob.glob(BASE_DIR + SPAM_DIR + '*.txt')
spam_design_matrix = generate_design_matrix(spam_filenames)
ham_filenames = glob.glob(BASE_DIR + HAM_DIR + '*.txt')
ham_design_matrix = generate_design_matrix(ham_filenames)
# Important: the test_filenames must be in numerical order as that is the
# order we will be evaluating your classifier
test_filenames = [BASE_DIR + TEST_DIR + str(x) + '.txt' for x in range(NUM_TEST_EXAMPLES)]
test_design_matrix = generate_design_matrix(test_filenames)

X = spam_design_matrix + ham_design_matrix
Y = np.array([1]*len(spam_design_matrix) + [0]*len(ham_design_matrix)).reshape((-1, 1)).squeeze()

np.savez(BASE_DIR + 'spam-data.npz', training_data=X, training_labels=Y, test_data=test_design_matrix)
