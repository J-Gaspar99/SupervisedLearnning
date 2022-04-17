import numpy as np

def tacnost_po_klasi(mat_konf, klase):
    tacnost_i = []
    N = mat_konf.shape[0]
    for i in range(N):
        j = np.delete(np.array(range(N)),i) 
        TP = mat_konf[i,i]
        F = 0
        F = (sum(mat_konf[i,j]) + sum(mat_konf[j,i]))
        TN = sum(sum(mat_konf)) - F - TP
        tacnost_i.append((TP+TN)/sum(sum(mat_konf)))
    tacnost_avg = np.mean(tacnost_i)
    return tacnost_avg

def osetljivost_po_klasi_makro(mat_konf, klase):
    osetljivost_i = []
    N = mat_konf.shape[0]
    for i in range(N):
        j = np.delete(np.array(range(N)),i) 
        TP = mat_konf[i,i]
        FN = sum(mat_konf[i,j])
        osetljivost_i.append(TP/(TP+FN))
    osetljivost_avg = np.mean(osetljivost_i)
    return osetljivost_avg

def specificnost_po_klasi(mat_konf, klase):
    specificnost_i = []
    N = mat_konf.shape[0]
    for i in range(N):
        j = np.delete(np.array(range(N)), i) 
        TP = mat_konf[i,i]
        F = 0
        F = (sum(mat_konf[i,j]) + sum(mat_konf[j,i]))
        FP = sum(mat_konf[j, i])
        TN = sum(sum(mat_konf)) - F - TP
        specificnost_i.append(TN/(TN+FP))
    specificnost_avg = np.mean(specificnost_i)
    return specificnost_avg

def preciznost_po_klasi_makro(mat_konf, klase):
    preciznost_i = []
    N = mat_konf.shape[0]
    for i in range(N):
        j = np.delete(np.array(range(N)), i) 
        TP = mat_konf[i, i]
        FP = sum(mat_konf[j, i])
        if TP != 0 and FP != 0:
            preciznost_i.append(TP/(TP+FP))
    print(TP, FP)
    preciznost_avg = np.mean(preciznost_i)
    return preciznost_avg

def osetljivost_po_klasi_mikro(mat_konf, klase):
    TP = 0
    FN = 0
    N = mat_konf.shape[0]
    for i in range(N):
        j = np.delete(np.array(range(N)),i) 
        TP += mat_konf[i,i]
        FN += sum(mat_konf[i,j])

    return TP/(TP+FN)

def preciznost_po_klasi_mikro(mat_konf, klase):
    TP = 0
    FP = 0
    N = mat_konf.shape[0]
    for i in range(N):
        j = np.delete(np.array(range(N)), i) 
        TP = mat_konf[i, i]
        FP = sum(mat_konf[j, i])
    return TP/(TP+FP)

def f_mera(preciznost, osetljivost):
    return 2 * preciznost * osetljivost / (preciznost + osetljivost)
