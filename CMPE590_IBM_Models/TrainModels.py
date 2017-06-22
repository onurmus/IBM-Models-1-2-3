from nltk.tokenize import word_tokenize
import IBMModel1
import IBMModel2
import IBMModel3
import string
import numpy as np
import Common


def create_tokenized_sentences(content_list, max_index):
    return_sentence_list = list()
    word_dictionary = {} #this dictionary will keep both word and its order in its language
    lang_order = 0
    cnt = 0
    max_len_sentence = 0
    translate_table = dict((ord(char), None) for char in string.punctuation)
    for row in content_list[:max_index]:
        if cnt == 0 :
            row = row.replace(u'\ufeff', '')
            cnt += 1

        row = row.translate(translate_table)
        tokens = word_tokenize(row.lower())

        if len(tokens) > max_len_sentence :
           max_len_sentence = len(tokens)

        produced_sentence = ""
        for token in tokens:
            if token not in word_dictionary:
                word_dictionary[token] = lang_order
                lang_order += 1
            produced_sentence = produced_sentence + token + " "
        produced_sentence = produced_sentence[:(len(produced_sentence) - 1)]  # remove last empty

        return_sentence_list.append(produced_sentence)

    return_sentence_list[0] = return_sentence_list[0].replace(u'\ufeff', '')  # ufeff character from document start
    return return_sentence_list, word_dictionary,max_len_sentence


def train_models():
    with open("Dictionary_files\BU_en.txt", encoding="utf8") as f:
        content_en = f.readlines()

    with open("Dictionary_files\BU_tr.txt", encoding="utf8") as f:
        content_tr = f.readlines()

    #just use sentences with length at most 10 words.
    new_content_en = list()
    new_content_tr = list()

    for sen_idx in range(len(content_en)):
        cur_en_sen = content_en[sen_idx].split()
        cur_tr_sen = content_tr[sen_idx].split()
        if len(cur_en_sen) < 11 and len(cur_tr_sen) < 11:
            new_content_en.append(content_en[sen_idx])
            new_content_tr.append(content_tr[sen_idx])

    content_en = new_content_en.copy()
    content_tr = new_content_tr.copy()


    max_num_of_translations = Common.num_of_train_sample

    # parse turksih sentences, tokenize the words
    turkish_sentences, turkish_word_dict, max_le = create_tokenized_sentences(content_tr, max_num_of_translations)

    # parse english sentences, tokenize the words
    english_sentences, english_word_dict, max_lf = create_tokenized_sentences(content_en, max_num_of_translations)

    print(content_en[145:150])
    print(content_tr[145:150])

    print(english_sentences[145:150])
    print(turkish_sentences[145:150])

    print(max_le)
    print(max_lf)

    '''
    np.save("models/e_word_dict",turkish_word_dict)
    np.save("models/f_word_dict",english_word_dict)

    t_e_f  = IBMModel1.expectation_maximization(turkish_word_dict,english_word_dict,turkish_sentences,english_sentences)
    np.save("models/t_e_f_mat_model1",t_e_f)

    t_e_f, a_i_le_lf_mat = IBMModel2.train(t_e_f,turkish_word_dict,english_word_dict,turkish_sentences,english_sentences,max_le,max_lf)
    np.save("models/t_e_f_mat_model2",t_e_f)
    np.save("models/a_i_le_lf_mat_model2",a_i_le_lf_mat)
    '''

    t_e_f = np.load('models/t_e_f_mat_model2.npy')
    a_i_le_lf_mat = np.load('models/a_i_le_lf_mat_model2.npy')
    t_e_f_mat, d_i_j_le_lf_mat, n_fi_f,p0,p1 = IBMModel3.train(t_e_f, a_i_le_lf_mat,turkish_word_dict,english_word_dict,turkish_sentences,english_sentences,max_le,max_lf)
    np.save("models/t_e_f_mat_model3",t_e_f_mat)
    np.save("models/d_i_j_le_lf_mat_model3",d_i_j_le_lf_mat)
    np.save("models/n_fi_f_mat_model3",n_fi_f)
    np.save("models/p0",p0)
    np.save("models/p1",p1)