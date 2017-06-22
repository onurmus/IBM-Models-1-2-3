import numpy as np
from datetime import datetime
import Common


def train(t_e_f_mat, e_word_dict,f_word_dict,e_sentences,f_sentences,max_le,max_lf):
    print("IBMModel2 Starts " + str(datetime.now()))
    a_i_le_lf_mat = np.zeros((max_lf, max_le, max_lf,max_le), dtype=float)

    for lf in range(max_lf):
        a_i_le_lf_mat[:,:,lf,:] = 1/(lf+1)

    num_of_e_word = len(e_word_dict)
    num_of_f_word = len(f_word_dict)

    t_e_f_mat_prev = np.full((num_of_e_word, num_of_f_word), 1,dtype=float)
    cnt_iter = 0

    print("While starts " + str(datetime.now()))
    while not Common.is_converged(t_e_f_mat,t_e_f_mat_prev,cnt_iter) :
        print(cnt_iter)
        cnt_iter += 1
        t_e_f_mat_prev = t_e_f_mat.copy()
        count_e_f = np.full((num_of_e_word, num_of_f_word), 0, dtype=float)
        total_f = np.full((num_of_f_word),0, dtype=float)
        count_a_i_le_lf = np.zeros((max_lf, max_le, max_lf,max_le), dtype=float)
        total_a_j_le_lf = np.zeros((max_le,max_le,max_lf),dtype=float)

        print("Sentence pair loop starts " + str(datetime.now()))
        for idx_e, e_sen in enumerate(e_sentences): #for all sentence pairs (e,f) do
            #le = length(e), lf = length(f)
            e_sen_words = e_sen.split(" ")
            f_sen_words = f_sentences[idx_e].split(" ")
            l_e = len(e_sen_words)
            l_f = len(f_sen_words)

            #compute normalization
            s_total = np.full((l_e),0,dtype=float)
            for j in range(l_e): #for j = 1 .. le do // all word positions in e
                s_total[j] = 0 #s-total(ej) = 0
                e_word = e_sen_words[j]
                for i in range(l_f): #for i = 0 .. lf do // all word positions in f
                    f_word = f_sen_words[i]
                    e_j = e_word_dict[e_word]
                    f_i = f_word_dict[f_word]
                    s_total[j] += t_e_f_mat[e_j][f_i] * a_i_le_lf_mat[i][j][l_f-1][l_e-1] #s-total(ej) += t(ej|fi) ∗ a(i|j,le,lf)
                #end for
            #end for

            #collect counts
            for j in range(l_e): #for j = 1 .. le do // all word positions in e
                e_word = e_sen_words[j]
                for i in range(l_f): #for i = 0 .. lf do // all word positions in f
                    f_word = f_sen_words[i]
                    e_j = e_word_dict[e_word]
                    f_i = f_word_dict[f_word]

                    c = t_e_f_mat[e_j][f_i] * a_i_le_lf_mat[i][j][l_f-1][l_e-1] / s_total[j] #c = t(ej|fi) ∗ a(i|j,le,lf) / s-total(ej)
                    count_e_f[e_j][f_i] += c #count(ej|fi) += c
                    total_f[f_i] += c #total(fi) += c
                    count_a_i_le_lf[i][j][l_f-1][l_e-1] += c #counta(i|j,le,lf) += c
                    total_a_j_le_lf[j][l_e-1][l_f-1] += c #totala(j,le,lf) += c
                #end for
            #end for
        #end for

        print("Estimate Probabilities starts " + str(datetime.now()))
        #estimate probabilities
        t_e_f_mat = np.full((num_of_e_word, num_of_f_word), 0,dtype=float) #t(e|f) = 0 for all e,f
        a_i_le_lf_mat = np.zeros((max_lf, max_le, max_lf,max_le), dtype=float) #a(i|j,le,lf) = 0 for all i,j,le,lf
        for f_idx in  range(num_of_f_word): #for all foreign words f do
            for e_idx in range(num_of_e_word): #for all English words e do
                if count_e_f[e_idx][f_idx] != 0 :
                    t_e_f_mat[e_idx][f_idx] = count_e_f[e_idx][f_idx] / total_f[f_idx]
            #end for
        #end for

        print("Estimating alignments starts " + str(datetime.now()))
        for i in range(max_lf):
            for  j in range(max_le):
                for le in range(max_le):
                    for lf in range(max_lf):
                        if count_a_i_le_lf[i][j][lf][le] != 0 :
                            a_i_le_lf_mat[i][j][lf][le] = count_a_i_le_lf[i][j][lf][le] / total_a_j_le_lf[j][le][lf]

    print("While loop ends print starts  " + str(datetime.now()))

    print(t_e_f_mat)
    print("IBMModel2 Ends " + str(datetime.now()))
    return t_e_f_mat, a_i_le_lf_mat


def get_translation_prob(e,f,t,a,e_dict,f_dict):
    const = Common.const
    l_e = len(e)
    l_f = len(f)
    res = const
    for j in range(l_e):
        e_word = e[j]
        if e_word in e_dict:
            e_j = e_dict[e_word]
        else:
            print(Common.O + "word '"+ e_word +"' is not found in target language dictionary" + Common.BL)
            continue
            #return 0

        sum = 0
        for i in range(l_f):
            f_word = f[i]

            if f_word in f_dict:
                f_i = f_dict[f_word]
                sum += t[e_j][f_i]*a[i][j][l_f-1][l_e-1]
            else:
                print(Common.B + "word '" + f_word  +"' is not found in source language dictionary"+ Common.BL)

        res *= sum

    if res == const:
        return 0
    return res