import numpy as np
import math
from datetime import datetime
import Common

def probability(a,e,f):
    fi_0 = len(e) - len(f) # buraya dikkat hata olabilir
    if fi_0 < 0:
        fi_0 = 0
    null_insert_prob = Common.nCr(len(e)-fi_0,fi_0) * (math.pow(p1, fi_0)) * (math.pow(p0, (len(e)-2*fi_0)))
    fertility_prob = 1
    for i in range(len(f)):
        f_word = f[i]
        f_i = f_word_dict[f_word]

        fertility = 0
        for j in range(len(e)):
            if i == a[j] : fertility += 1
        fertility_prob *= Common.factorial(fertility) *n_fi_f[fertility][f_i]

    lexical_distortion_prob = 1
    for j in range(len(e)):
        e_word = e[j]
        f_word = f[a[j]]
        e_j = e_word_dict[e_word]
        f_i = f_word_dict[f_word]
        lexical_distortion_prob *= t_e_f_mat[e_j][f_i] * d_i_j_le_lf_mat[a[j]][j][len(f)-1][len(e)-1]

    return null_insert_prob * fertility_prob * lexical_distortion_prob


def neighboring(a,j_pegged,e_words,f_words):
    N = []
    l_f = len(f_words)
    l_e = len(e_words)
    #N.append(a) # bu satırı ben ekledim.
    for neg_j in range(l_e):
        if neg_j != j_pegged:
            for neg_i in range(-1,l_f):
                # !!!!!!!!!!! buraya neg_i != a[j_pegged]= i kontrolü gelmeli mi !!!!!
                neg_a = a.copy()
                neg_a[neg_j] = neg_i
                N.append(neg_a.copy())

    for j_1 in range(l_e):
        if j_1 != j_pegged:
            for j_2 in range(l_e):
                if j_2 != j_pegged and j_2 != j_1 :
                    neg_a = a.copy()
                    temp = neg_a[j_1]
                    neg_a[j_1] = neg_a[j_2]
                    neg_a[j_2] = temp
                    N.append(neg_a.copy())
    return N


def hillclimb(a,j_pegged,e_words,f_words):
    a_old = []
    while a != a_old:
        a_old = a.copy()
        for a_neighbor in neighboring(a,j_pegged,e_words,f_words):
            if probability(a_neighbor,e_words,f_words) > probability(a,e_words,f_words):
                a = a_neighbor.copy()
    return a


def sample(e_words, f_words):
    A = []
    a = []
    l_f = len(f_words)
    l_e = len(e_words)
    for j in range(l_e):
        a.append(-1)

    for j in range(l_e):
        for i in range(-1,l_f):
            a[j] = i
            for neg_j in range(l_e):
                if neg_j != j:
                    #a(j') = argmaxi' t(ej' |fi' ) d(i'|j', length(e), length(f))
                    argmaxi = 0
                    for neg_i in range(-1,l_f):
                        #if neg_i != i:
                        f_word = f_words[neg_i]
                        e_word = e_words[neg_j]
                        e_j = e_word_dict[e_word]
                        f_i = f_word_dict[f_word]
                        temp = t_e_f_mat[e_j][f_i] * d_i_j_le_lf_mat[neg_i][neg_j][l_f-1][l_e-1]
                        if temp > argmaxi:
                            argmaxi = temp
                            a[neg_j] = neg_i
            #end for
            a = hillclimb(a,j,e_words,f_words)
            N = neighboring(a,j,e_words,f_words)
            #!!!!!!!!!!!!!!!!burada a da sample set e eklenmeli mi? !!!!!!!!!!!
            if N != []:
                for n in N:
                    #if not n in A:
                    A.append(n)
        #end for
    #end for
    return A


def train(arg_t_e_f_mat, arg_d_i_le_lf_mat, arg_e_word_dict,arg_f_word_dict,e_sentences,f_sentences,max_le,max_lf):
    print("IBMModel3 Starts " + str(datetime.now()))
    global t_e_f_mat, d_i_j_le_lf_mat, n_fi_f, e_word_dict,f_word_dict, p0, p1

    t_e_f_mat = arg_t_e_f_mat
    d_i_j_le_lf_mat = arg_d_i_le_lf_mat
    e_word_dict = arg_e_word_dict
    f_word_dict = arg_f_word_dict
    p0 = 0.5
    p1 = 0.5

    num_of_e_word = len(e_word_dict)
    num_of_f_word = len(f_word_dict)
    max_fertility = 20

    n_fi_f =  np.full((max_fertility, num_of_f_word), 1/max_fertility, dtype=float) #φ(n|f) = 0 for all f,n

    t_e_f_mat_prev = np.full((num_of_e_word, num_of_f_word), 1,dtype=float)
    cnt_iter = 0

    print("While starts " + str(datetime.now()))
    while not Common.is_converged(t_e_f_mat,t_e_f_mat_prev,cnt_iter) :
        print(cnt_iter)
        cnt_iter += 1
        t_e_f_mat_prev = t_e_f_mat.copy()
        #set all count∗ and total∗ to 0
        count_t = np.full((num_of_e_word, num_of_f_word), 0, dtype=float)
        total_t = np.full((num_of_f_word),0, dtype=float)
        count_d = np.zeros((max_le,max_lf, max_lf,max_le), dtype=float)
        total_d = np.zeros((max_lf, max_lf,max_le),dtype=float)
        count_f = np.full((max_fertility, num_of_f_word), 0, dtype=float)
        total_f = np.full((num_of_f_word),0, dtype=float)
        count_p0 = 0
        count_p1 = 0

        for idx_e, e_sen in enumerate(e_sentences): #for all sentence pairs (e,f) do
            print("we are in sentence " + str(idx_e) + "  " + str(datetime.now()))
            #le = length(e), lf = length(f)
            e_sen_words = e_sen.split(" ")
            f_sen_words = f_sentences[idx_e].split(" ")
            l_e = len(e_sen_words)
            l_f = len(f_sen_words)
            A = sample(e_sen_words,f_sen_words)
            total_null = 0
            # collect counts
            c_total = 0
            for a in A: #for all a ∈ A do ctotal += probability( a, e, f );
                c_total += probability(a,e_sen_words,f_sen_words)


            for a in A: #for all a ∈ A do
                c = probability(a,e_sen_words,f_sen_words) /c_total #c = probability( a, e, f ) / ctotal
                total_null = 0
                for j in range(l_e): #for j = 0 .. length(f) do
                    e_word = e_sen_words[j]
                    f_word = f_sen_words[a[j]]
                    e_j = e_word_dict[e_word]
                    f_i = f_word_dict[f_word]
                    count_t[e_j][f_i] += c              # lexical translation
                    total_t[f_i] += c                   # lexical translation
                    count_d[j][a[j]][l_f-1][l_e-1] += c #distortion
                    total_d[a[j]][l_f-1][l_e-1] += c    #distortion
                    if a[j] == -1: total_null += 1       #if a(j) == 0 then null++; // null insertion
                #end for

                #countp1 += null ∗ c; countp0 += (length(e) - 2 ∗ null) ∗ c
                count_p1 += total_null * c
                count_p0 += (l_e-2*total_null)*c

                for i in range(l_f):
                    fertility = 0
                    for j in range(l_e):
                        if i == a[j]: fertility += 1
                    #end for
                    f_word = f_sen_words[i]
                    f_i = f_word_dict[f_word]
                    count_f[fertility][f_i] += c
                    total_f[f_i] += c
                #end for
            #end for
        #end for

        #estimate probability distribution
        t_e_f_mat = np.full((num_of_e_word, num_of_f_word), 0,dtype=float) #t(e|f) = 0 for all e,f
        d_i_j_le_lf_mat = np.zeros((max_lf, max_le, max_lf,max_le), dtype=float) #d(j|i,le,lf) = 0 for all i,j,le,lf
        n_fi_f =  np.full((max_fertility, num_of_f_word), 0, dtype=float) #φ(n|f) = 0 for all f,n

        #for all (e,f) in domain( countt ) do t(e|f) = countt(e|f) / totalt(f)
        for f_idx in  range(num_of_f_word): #for all foreign words f do
            for e_idx in range(num_of_e_word): #for all English words e do
                if count_t[e_idx][f_idx] != 0 :
                    t_e_f_mat[e_idx][f_idx] = count_t[e_idx][f_idx] / total_t[f_idx]
            #end for
        #end for

        #for all (i,j,le,lf) in domain( countd ) do
        for i in range(max_lf):
            for  j in range(max_le):
                for le in range(max_le):
                    for lf in range(max_lf):
                        if count_d[j][i][lf][le] != 0 :
                            d_i_j_le_lf_mat[i][j][lf][le] = count_d[j][i][lf][le] / total_d[i][lf][le]

        for fi in range(max_fertility):
            for j in range(num_of_f_word):
                if count_f[fi][j] != 0:
                    n_fi_f[fi][j] = count_f[fi][j] / total_f[j]

        p1 = count_p1 / (count_p1 + count_p0)
        p0 = 1-p1
        print(t_e_f_mat)

    print("While loop ends print starts  " + str(datetime.now()))


    print("IBMModel3 Ends " + str(datetime.now()))
    return t_e_f_mat, d_i_j_le_lf_mat,n_fi_f,p0,p1


def get_translation_prob(e,f,t,a,n_fi_f,e_dict,f_dict):
    const = Common.const
    l_e = len(e)
    l_f = len(f)
    res = const
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
                sum += t[e_j][f_i]*a[i][j][l_f-1][l_e-1]*n_fi_f[i][f_i]
            else:
                print("word '" + f_word  +"' is not found in source language dictionary")

        res *= sum

    return res