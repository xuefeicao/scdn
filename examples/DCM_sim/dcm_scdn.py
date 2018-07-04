"""
An example with 100 subjects fMRI signals with same stimuli. Date was generated from DCM model.
Data are included in the folder data/
"""
import sys
sys.path.append('../..')
#import scdn 
from scdn.scdn_analysis import scdn_multi_sub
from scdn.evaluation import eva 

folder_name = ['Analysis/' + str(i) + '/' for i in range(1, 101)]
data_file = ['data/fMRI_' + str(i) + '.txt' for i in range(1, 101)]
real_parameter_file = 'data/real.pkl' # dict for real para
stimuli_folder = ['data/stimuli/']*100 # This folder only contains evt files, two subjects share same stimuli
dt = 3 #TR of fMRI
scdn_multi_sub(folder_name, data_file, stimuli_folder, val_pair=(0,1), dt=dt, lam=[0.1,0.01,1], mu=[0.1, 1, 5, 10, 100], mu_1=[1], mu_2=[1e-4, 1e-2, 1], lam_1=[0], tol=1e-2, max_iter=100, N=50, fold=0.5, share_stimuli=True, num_cores=2, init=False, B_u=False)
eva(folder_name, real_parameters=real_parameter_file)
