import numpy as np
from datetime import datetime
import math
import Common


def expectation_maximization(turkish_word_dict,english_word_dict,turkish_sentences,english_sentences):
    num_of_tur_word = len(turkish_word_dict)
    num_of_eng_word = len(english_word_dict)
    # em algorithm
    t_e_f_mat = np.full((len(turkish_word_dict), len(english_word_dict)), 1 / len(english_word_dict),dtype=float)
    t_e_f_mat_prev = np.full((len(turkish_word_dict), len(english_word_dict)), 1,dtype=float)

    cnt_iter = 0
    while not Common.is_converged(t_e_f_mat,t_e_f_mat_prev,cnt_iter) :
        print(cnt_iter)
        cnt_iter += 1
        t_e_f_mat_prev = t_e_f_mat.copy()
        count_e_f = np.full((len(turkish_word_dict), len(english_word_dict)), 0, dtype=float)
        total_f = np.full((len(english_word_dict)),0, dtype=float)
        print("sentece pair giris")
        for idx_tur, tur_sen in enumerate(turkish_sentences): #for all sentence pairs (e,f) do
            #compute normalization
            tur_sen_words = tur_sen.split(" ")
            s_total = np.full((len(tur_sen_words)),0,dtype=float)
            for idx_word in range(len(tur_sen_words)): #for all words e in e do
                tur_word = tur_sen_words[idx_word]
                s_total[idx_word] = 0
                eng_sen_words = english_sentences[idx_tur].split(" ")
                for eng_word in eng_sen_words: #for all words f in f do
                    idx_tur_in_dict =turkish_word_dict[tur_word]
                    idx_eng_in_dict = english_word_dict[eng_word]
                    s_total[idx_word] += t_e_f_mat[idx_tur_in_dict][idx_eng_in_dict]
                #end for
            #end for

            #collect counts
            tur_sen_words = tur_sen.split(" ")
            for idx_word in range(len(tur_sen_words)): #for all words e in e do
                tur_word = tur_sen_words[idx_word]
                eng_sen_words = english_sentences[idx_tur].split(" ")
                for eng_word in eng_sen_words: #for all words f in f do
                    idx_tur_in_dict =turkish_word_dict[tur_word]
                    idx_eng_in_dict = english_word_dict[eng_word]
                    count_e_f[idx_tur_in_dict][idx_eng_in_dict] += t_e_f_mat[idx_tur_in_dict][idx_eng_in_dict] / s_total[idx_word]
                    total_f[idx_eng_in_dict] += t_e_f_mat[idx_tur_in_dict][idx_eng_in_dict] / s_total[idx_word]
                #end for
            #end for
        #end for

        print("ucuncu for loop giris ")
        print(str(datetime.now()))
        #estimate probabilities
        for eng_idx in  range(num_of_eng_word): #for all foreign words f do
            for tur_idx in range(num_of_tur_word): #for all English words e do
                if count_e_f[tur_idx][eng_idx] != 0 :
                    t_e_f_mat[tur_idx][eng_idx] = count_e_f[tur_idx][eng_idx] / total_f[eng_idx]
            #end for
        #end for

        print("finish ")
        print(str(datetime.now()))
    #end while

    print(t_e_f_mat)
    print(cnt_iter)

    return t_e_f_mat


def get_translation_prob(e,f,t,e_dict,f_dict):
    const = Common.const
    l_e = len(e)
    l_f = len(f)
    res = const / math.pow((l_f+1),l_e)
    for j in range(l_e):
        e_word = e[j]
        if e_word in e_dict:
            e_j = e_dict[e_word]
        else:
            print("word '"+ e_word +"' is not found in target language dictionary")
            continue
            #return 0

        sum = 0
        for i in range(l_f):
            f_word = f[i]

            if f_word in f_dict:
                f_i = f_dict[f_word]
                sum += t[e_j][f_i]
            else:
                print("word '" + f_word  +"' is not found in source language dictionary")

        res *= sum

    return res