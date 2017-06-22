import numpy as np
from nltk.tokenize import word_tokenize
import string
import IBMModel1
import IBMModel2
import IBMModel3
import Common


def get_tokens_of_sentence(sentence):
    translate_table = dict((ord(char), None) for char in string.punctuation)
    sentence = sentence.translate(translate_table)
    tokens = word_tokenize(sentence.lower())
    return tokens


def sentence_tester(sentence):
    try:
        num_of_sentences = int(input("\nHow many possible results you want to supply for sentece '"+ sentence.strip() +"': \n"))
    except ValueError:
        print ("Not a number")

    possible_sentences = list()
    for i in range(num_of_sentences):
        input_sentence = input("Type possible sentence number " + str((i+1)) + " : ")
        possible_sentences.append(input_sentence)

    f_sentence = get_tokens_of_sentence(sentence)

    max_score = -1
    max_sentence = ""
    for poss_sentence in possible_sentences:
        e_sentence = get_tokens_of_sentence(poss_sentence)

        if model_number == 1: #IBM Model 1
            prob = IBMModel1.get_translation_prob(e_sentence,f_sentence,t_e_f,e_word_dict,f_word_dict)
            print(Common.P + "probability for sentence '" + poss_sentence + "' is " + str(prob) + Common.BL)
        elif model_number == 2: #IBM Model 2
            prob = IBMModel2.get_translation_prob(e_sentence,f_sentence,t_e_f,a_i_j,e_word_dict,f_word_dict)
            print(Common.P + "probability for sentence '" + poss_sentence + "' is " + str(prob) + Common.BL)
        elif model_number == 3: #IBM Model 3
            prob = IBMModel3.get_translation_prob(e_sentence,f_sentence,t_e_f,a_i_j,n_fi_f,e_word_dict,f_word_dict)
            print(Common.P + "probability for sentence '" + poss_sentence + "' is " + str(prob) + Common.BL)

        if prob > max_score:
            max_score = prob
            max_sentence = poss_sentence

    print(Common.R + "tranlation result is '" + max_sentence +"'  with probability : " + str(max_score) + Common.BL)

def test(arg_model_number, is_sentence_translate,sentence_to_translate):
    global t_e_f, a_i_j, n_fi_f, e_word_dict,f_word_dict,content_f,model_number

    model_number = arg_model_number

    if model_number == 1: #IBM Model 1
        t_e_f = np.load('models/t_e_f_mat_model1.npy')
    elif model_number == 2: #IBM Model 2
        t_e_f = np.load('models/t_e_f_mat_model2.npy')
        a_i_j = np.load('models/a_i_le_lf_mat_model2.npy')
    elif model_number == 3: #IBM Model 3
        t_e_f = np.load('models/t_e_f_mat_model3.npy')
        a_i_j = np.load('models/d_i_j_le_lf_mat_model3.npy')
        n_fi_f = np.load('models/n_fi_f_mat_model3.npy')

    e_word_dict = np.load("models/e_word_dict.npy").item()
    f_word_dict = np.load("models/f_word_dict.npy").item()

    if is_sentence_translate:
        sentence_tester(sentence_to_translate)
    else:
        with open("Dictionary_files\BU_en.txt", encoding="utf8") as f:
                content_f = f.readlines()

        new_content_f = list()

        for sen_idx in range(len(content_f)):
            cur_f_sen = content_f[sen_idx].split()
            if len(cur_f_sen) < 11:
                new_content_f.append(content_f[sen_idx])

        content_f = new_content_f.copy()

        for sentence in content_f [Common.num_of_train_sample:(Common.num_of_train_sample + Common.num_of_test_sample)]:
            sentence_tester(sentence)


