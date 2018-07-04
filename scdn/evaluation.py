from __future__ import print_function
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import math
import six 
from six.moves import cPickle as pkl

def fd_0(A1,A):
    """
    compute AUC of estimated A if real A is known, used for simulated data

    Parameters
    ----------
    A1: numpy array, real A
    A: estimated data, all estimations from all subjects

    Returns
    ----------
    scalar, AUC
    """
    if np.sum(abs(A1)) < 1e-6:
        return -1
    if len(A1.shape) == 3 and np.sum(abs(A1)>0) == (A1.shape[0]*A1.shape[1]*A1.shape[2]):
        return -1
    if len(A1.shape) == 2 and np.sum(abs(A1)>0) == (A1.shape[0]*A1.shape[1]):
        return -1
    if len(A.shape) == 3:
        tmp = abs(np.mean(A,axis=2))
        
    else:
        tmp = abs(np.mean(A,axis=3))
    if np.max(tmp) == 0:
        return -1
    tmp = tmp/np.max(tmp)
    A1 = (abs(A1)>0)
    sr = roc_auc_score(A1.reshape((-1)),tmp.reshape((-1)))
    return sr

def fd_1(A1,A):
    """
    compute AUC of estimated A if real A is known, used for simulated data, based on number of nonzero trials of one entry. 

    Parameters
    ----------
    A1: numpy array, real A
    A: estimated data, all estimations from all subjects

    Returns
    ----------
    scalar, AUC
    """
    if np.sum(abs(A1))<1e-6:
        return -1
    if len(A1.shape)==3 and np.sum(abs(A1)>0)==(A1.shape[0]*A1.shape[1]*A1.shape[2]):
        return -1
    if len(A1.shape)==2 and np.sum(abs(A1)>0)==(A1.shape[0]*A1.shape[1]):
        return -1
    n=A.shape[-1]
    m1=A.shape[0]
    m2=A.shape[1]
    A1=(abs(A1)>0)
    A=abs(A)
    if len(A.shape)==3:
        tmp=np.zeros((m1,m2))
        for i in range(m1):
            for j in range(m2):
                tmp[i,j]=sum(abs(A[i,j,:])>0)
    else:
        tmp=np.zeros((m1,m2,A.shape[2]))
        for i in range(m1):
            for j in range(m2):
                for k in range(A.shape[2]):
                    tmp[i,j,k]=sum(abs(A[i,j,k,:])>0)
    tmp=1.0*tmp/n
    sr=roc_auc_score(A1.reshape((-1)),tmp.reshape((-1)))
    return sr


def eva(folder_name, real_parameters=None, num_iterations=10000, alpha=0.1):
    """
    evaluation of estimations

    Parameters
    -----------
    folder_name: folder names for all subjects analysis, the same as meaning as that in function CDN_multi_sub
    nums_iterations, alpha: bootstrap para
    """
    n = len(folder_name)

    for i in range(n):
        with open(folder_name[i]+'results/result.pkl', 'rb') as f:
            if six.PY2:
                save = pkl.load(f)
            else:
                save = pkl.load(f, encoding='latin1')
        A = save['A']
        B = save['B']
        C = save['C']
        if i == 0:
            A_all = np.zeros((A.shape[0], A.shape[1], n))
            B_all = np.zeros((B.shape[0], B.shape[1], B.shape[2], n))
            C_all = np.zeros((C.shape[0], C.shape[1], n))
        A_all[:,:,i] = A
        B_all[:,:,:,i] = B
        C_all[:,:,i] = C
    if real_parameters:
        with open(real_parameters, 'rb') as f:
            if six.PY2:
                save = pkl.load(f)
            else:
                save = pkl.load(f, encoding='latin1')
        A_real = save['A_real']
        B_real = save['B_real']
        C_real = save['C_real']

        auc_a = fd_0(A_real, A_all)
        auc_b = fd_0(B_real, B_all)
        auc_c = fd_0(C_real, C_all)
        print('AUC(A):{0}, AUC(B):{1}, AUC(C):{2}'.format(auc_a, auc_b, auc_c))
        
        auc_a = fd_1(A_real, A_all)
        auc_b = fd_1(B_real, B_all)
        auc_c = fd_1(C_real, C_all)
        print('AUC(A):{0}, AUC(B):{1}, AUC(C):{2}'.format(auc_a, auc_b, auc_c))


    









