import os
import numpy as np
from scipy.stats import norm
import scipy.stats
import re
from numpy import asarray
from scipy.spatial import distance
from tensorboard.backend.event_processing import event_accumulator
pattern1 = r".*_\D*(\d+).npz"

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth
def extrac_acc_from_ea(path):
    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()
    ea.Tags()  # 查看数据文件中的数据标签
    choice='False' #'True '
    acces_item1 = ea.scalars.Items('Test_backdoor_{}/Accuracy_Top-1'.format(choice))  # 查看指定的标量数据
    access1 = []
    for i, item in enumerate(acces_item1):
        access1.append(item.value)
    choice='True '
    acces_item2 = ea.scalars.Items('Test_backdoor_{}/Accuracy_Top-1'.format(choice))  # 查看指定的标量数据
    access2 = []
    for i, item in enumerate(acces_item2):
        access2.append(item.value)
    return access1, access2

def extract_acc_loss(name: str = ''):
    patternAcc = r"(?<=Top-1:)\s*\d*\.\d*"
    patternLoss = r"(?<=value:)\s*\d*\.\d*"
    with open(name) as f:
        datafile = f.readlines()
        accList = []
        backdoor_accList = []
        for line in datafile:
            if 'Backdoor False' in line:
                acc = re.findall(patternAcc, line)
                loss = re.findall(patternLoss, line)
                if len(loss) == 0:
                    loss = [5]
                accList.append(float(acc[0]))
            if 'Backdoor True' in line:
                backdoor_acc = re.findall(patternAcc, line)
                backdoor_accList.append(float(backdoor_acc[0]))
    f.close()
    return accList, backdoor_accList
def sort_key(s):
    # 排序关键字匹配
    # 匹配开头数字序号
    if s:
        try:
            c = re.findall('\d+', s)[0]
        except:
            c = -1
        return int(c)


def get_new_clients(oldclients):
    newclients = []
    for i in oldclients:
        if '.npz' in i:
            newclients.append(i)
    return newclients


def get_new_clients_epoch(oldclients, epoch):
    newclients = []
    for i in oldclients:
        if ('.npz' in i) and ('iteration' + str(epoch) + '_' in i):
            newclients.append(i)
    return newclients


def get_name_flag(name, epoch):
    if 'mult' in name:
        flag = str(epoch) + '_'
    else:
        flag = 'epoch' + str(epoch) + '_'
    return flag


def get_clients_deletewhite(oldclients, white):
    newclients = []
    for i in oldclients:
        if '.npz' in i:
            i_id = re.findall(pattern1, i)[0]
            if i_id not in white:
                newclients.append(i)
    return newclients


def get_clients_epoch_deletewhite(oldclients, white, epoch):
    newclients = []
    for i in oldclients:
        if ('.npz' in i) and ('iteration' + str(epoch) + '_' in i):
            i_id = re.findall(pattern1, i)[0]
            if i_id not in white:
                newclients.append(i)
    return newclients


def make_dir(name):
    try:
        os.mkdir(name)
    except FileNotFoundError:
        os.makedirs(name)
    except FileExistsError:
        print(f'{name} have been built')


def MadScore(distance):
    median = np.median(distance)
    abs_dev = distance - median
    mad = np.median(abs(abs_dev))  # *b
    mod_z_score = norm.ppf(0.75) * abs_dev / mad
    # result = mod_z_score>3
    return mod_z_score  # result
def MadScore_mult(distance):
    median = np.median(distance,axis=0)
    abs_dev = distance - median
    mad = np.median(abs(abs_dev),axis=0)  # *b
    mod_z_score = norm.ppf(0.75) * abs_dev / mad
    # result = mod_z_score>3
    mod_z_score[np.isnan(mod_z_score)] = 0
    return mod_z_score  # result

def is_pos_def(A):
    if np.allclose(A, A.T):  # check if A is Symmetric Matrices
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False


def CovMatrix(data):
    covariance_matrix = np.cov(data, rowvar=False)
    if is_pos_def(covariance_matrix):
        inv_covariance_matrix = np.linalg.inv(covariance_matrix)
        if is_pos_def(inv_covariance_matrix):
            return covariance_matrix, inv_covariance_matrix
        else:
            print("Error: Inverse of Covariance Matrix is not positive definite!")
    else:
        print("Error: Covariance Matrix is not positive definite!")


def MahalanobisDist(inv_cov_matrix, mean_distr, data, verbose=False):
    inv_covariance_matrix = inv_cov_matrix
    vars_mean = mean_distr
    diff = data - vars_mean
    md = []
    for i in range(len(diff)):
        md.append(np.sqrt(diff[i].dot(inv_covariance_matrix).dot(diff[i])))
    return md


def MD_threshold(dist, extreme=False, verbose=False):
    k = 3. if extreme else 2.
    threshold = np.mean(dist) * k
    return threshold


def spearman_footrule_distance(s, t):
    """
    Computes the Spearman footrule distance between two full lists of ranks:
        F(s,t) = sum[ |s(i) - t(i)| ]/S,
    the normalized sum over all elements in a set of the absolute difference between
    the rank according to s and t.  As defined, 0 <= F(s,t) <= 1.
    S is a normalizer which is equal to 0.5*len(s)^2 for even length ranklists and
    0.5*(len(s)^2 - 1) for odd length ranklists.
    If s,t are *not* full, this function should not be used. s,t should be array-like
    (lists are OK).
    """
    s = s.ravel()
    t = t.ravel()
    # check that size of intersection = size of s,t?
    assert len(s) == len(t)
    sdist = sum(abs(asarray(s) - asarray(t)))
    # c will be 1 for odd length lists and 0 for even ones
    # c = len(s) % 2
    # normalizer = 0.5*(len(s)**2 - c)
    normalizer = len(s)
    return sdist / normalizer
def spearman_footrule_distance_multi(s, t):
    """
    Computes the Spearman footrule distance between two full lists of ranks:
        F(s,t) = sum[ |s(i) - t(i)| ]/S,
    the normalized sum over all elements in a set of the absolute difference between
    the rank according to s and t.  As defined, 0 <= F(s,t) <= 1.
    S is a normalizer which is equal to 0.5*len(s)^2 for even length ranklists and
    0.5*(len(s)^2 - 1) for odd length ranklists.
    If s,t are *not* full, this function should not be used. s,t should be array-like
    (lists are OK).
    """
    if len(s.shape) == 3:
        results=np.sum((s - t).reshape(s.shape[0],s.shape[1]*s.shape[2]),axis=1)/s[0].size
    else:
        results = np.sum((s - t),axis=1) / s.shape[1]
    return results

# ref: https://stackoverflow.com/questions/5101004/python-code-for-earth-movers-distance/10670028
def emd(a, b):
    a = a.ravel()
    b = b.ravel()
    earth = 0
    earth1 = 0
    diff = 0
    s = len(a)
    su = []
    diff_array = []
    for i in range(0, s):
        diff = a[i] - b[i]
        diff_array.append(diff)
        diff = 0
    for j in range(0, s):
        earth = (earth + diff_array[j])
        earth1 = abs(earth)
        su.append(earth1)
    emd_output = sum(su) / (s - 1)
    return emd_output


def adjcos(mat1, mat2):
    mat1 = mat1.astype(np.float64)
    mat2 = mat2.astype(np.float64)
    v1 = mat1.ravel()
    v2 = mat2.ravel()
    v1v2mean = (np.mean(v1) + np.mean(v2)) / 2
    # v1v2mean = 4.5
    v1 -= v1v2mean
    v2 -= v1v2mean
    norm1 = np.sqrt(v1.dot(v1))
    norm2 = np.sqrt(v2.dot(v2))
    return v1.dot(v2) / norm1 / norm2

def adjcos_mul(mat1, mat2):
    # mat1 = mat1.astype(np.float64)
    # mat2 = mat2.astype(np.float64)
    mat1 = np.array(mat1,dtype=float)
    mat2 = np.array(mat2,dtype=float)
    v1 = mat1.reshape(mat1.shape[0],-1)
    v2 = mat2.reshape(mat2.shape[0],-1)
    v1v2mean = (np.mean(v1,axis=1) + np.mean(v2,axis=1)) / 2
    # v1v2mean = 4.5
    v1 = v1-np.repeat(v1v2mean.reshape(v1v2mean.shape[0],1),v1.shape[1],axis=1)
    v2 = v2-np.repeat(v1v2mean.reshape(v1v2mean.shape[0],1),v2.shape[1],axis=1)
    norm1 = np.sqrt(np.sum(np.multiply(v1,v1),axis=1))
    norm2 = np.sqrt(np.sum(np.multiply(v2,v2),axis=1))
    result = np.sum(np.multiply(v1,v2),axis=1) / norm1 / norm2
    return result

# Normalized correlation (cosine similarity)
def corr(mat1, mat2):
    # xpy = cupy.get_array_module(mat1)
    mat1 = mat1.astype(np.float64)
    mat2 = mat2.astype(np.float64)
    v1 = mat1.ravel()
    v2 = mat2.ravel()
    v1 -= v1.mean()
    v2 -= v2.mean()
    norm1 = np.sqrt(v1.dot(v1))
    norm2 = np.sqrt(v2.dot(v2))

    if norm1*norm2 == 0 :
        return -1
    return v1.dot(v2) / norm1 / norm2

def corr_mul(mat1, mat2):
    # xpy = cupy.get_array_module(mat1)
    mat1 = np.array(mat1, dtype=float)
    mat2 = np.array(mat2, dtype=float)
    v1 = mat1.reshape(mat1.shape[0], -1)
    v2 = mat2.reshape(mat2.shape[0], -1)
    v1mean= np.mean(v1, axis=1)
    v1 = v1 - np.repeat(v1mean.reshape(v1mean.shape[0], 1), v1.shape[1], axis=1)
    v2mean = np.mean(v2, axis=1)
    v2 = v2 - np.repeat(v2mean.reshape(v2mean.shape[0], 1), v2.shape[1], axis=1)
    norm1 = np.sqrt(np.sum(np.multiply(v1,v1),axis=1))
    norm2 = np.sqrt(np.sum(np.multiply(v2,v2),axis=1))
    results = np.sum(np.multiply(v1, v2), axis=1) / norm1 / norm2
    results[np.isnan(results)] = -1
    return results
# Numberic cross-correlation
def ncorr2(mat1, mat2):
    # xpy = cupy.get_array_module(mat1) #numpy or cupy
    mat1 -= mat1.mean()
    mat2 -= mat2.mean()
    mat2 = mat2[::-1, ::-1]
    f1 = np.fft.fft2(mat1)
    f2 = np.fft.fft2(mat2)
    return np.real(np.fft.ifft2(f1 * f2))


"""
calculate PCE between mat X and Y.
nsize: neighborhood size around the peak.
searchRange: size of rectangular searching range for shifting.
reutrns (PCE,peak,peakx,peaky) where peakx,peaky indicates the shifting vector.
X/Y: on host or device memory
"""


def PCE(X, Y, nsize=1, searchRange=[1, 1]):
    # xpy = cupy.get_array_module(X) #numpy or cupy
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    m, n = X.shape
    X -= X.mean()
    Y -= Y.mean()
    XY = ncorr2(X, Y)

    searchMargin = (searchRange[0] // 2, searchRange[1] // 2)
    XYshifted = np.roll(XY, (m // 2 + 1, n // 2 + 1), axis=(0, 1))
    XYinRange = XYshifted \
        [m // 2 - searchMargin[0]:m // 2 + searchMargin[0] + 1, n // 2 - searchMargin[1]:n // 2 + searchMargin[1] + 1]
    absXYinRange = np.abs(XYinRange)
    (peaki, peakj) = np.unravel_index(np.argmax(absXYinRange), absXYinRange.shape)

    sgn = np.sign(XYinRange[peaki, peakj])
    peaky = searchMargin[0] - peaki
    peakx = searchMargin[1] - peakj
    XYpeak = XYinRange[peaki, peakj]

    margin = nsize // 2
    neighbor_energy = np.sum( \
        XYshifted[m // 2 + peaky - margin:m // 2 + peaky + margin + 1,
        n // 2 + peakx - margin:n // 2 + peakx + margin + 1] ** 2)

    PCE_energy = (np.sum(XY ** 2) - neighbor_energy) / (m * n - (2 * margin + 1) ** 2)
    pce = sgn * XYpeak ** 2 / PCE_energy
    # if xpy==cupy:
    #     return tuple(map(to_host_scalar,[pce,XYpeak,peakx,peaky]))
    # else:
    return (pce, XYpeak, peakx, peaky)

def adjcorr(v1, v2):
    v1 -= v1.mean()
    v2 -= v2.mean()
    v1v2mean = (np.mean(v1) + np.mean(v2)) / 2
    # v1v2mean = 4.5
    v1 -= v1v2mean
    v2 -= v1v2mean


    norm1 = np.sqrt(v1.dot(v1))
    norm2 = np.sqrt(v2.dot(v2))

    if norm1 * norm2 == 0:
        return -100
    return v1.dot(v2) / norm1 / norm2
def norm255(mat):
    new_mat = np.uint8(255 * (mat - np.min(mat)) / (np.max(mat) - np.min(mat)))
    return new_mat


def norm1(mat):
    new_mat = (mat - np.min(mat)) / (np.max(mat) - np.min(mat))
    return new_mat


def get_sim(sim_method, img, imgs_mean):
    if 'PCE' in sim_method:
        (sim, _, _, _) = PCE(img, imgs_mean, nsize=11, searchRange=[3, 3])
    elif sim_method == 'corr':
        sim = corr(img, imgs_mean)
    elif sim_method == 'corr_mul':
        sim = corr_mul(img, imgs_mean)
    elif sim_method == 'adjcos':
        sim = adjcos(img, imgs_mean)
    elif sim_method == 'adjcos_mul':
        sim = adjcos_mul(img, imgs_mean)
    elif sim_method == 'emd':
        sim = emd(img, imgs_mean)
    elif sim_method == 'footrule':
        sim = spearman_footrule_distance(img, imgs_mean)
    elif sim_method == 'footrule_mult':
        sim = spearman_footrule_distance_multi(img, imgs_mean)
    elif sim_method == 'mean':
        sim = np.mean(img)
    elif 'pear' in sim_method:
        sim = scipy.stats.pearsonr(img.ravel(), imgs_mean.ravel())[0]
    elif 'sp' in sim_method:
        sim = scipy.stats.spearmanr(img.ravel(), imgs_mean.ravel())[0]
    elif 'ken' in sim_method:
        sim = scipy.stats.kendalltau(img.ravel(), imgs_mean.ravel())[0]
    elif 'euclidean' in sim_method:
        sim=distance.euclidean(img.ravel(), imgs_mean.ravel())
    elif 'adjcorr' in sim_method:
        sim=adjcorr(img.ravel(), imgs_mean.ravel())
    return sim
def get_threshold(values,n_std):
    std = np.std(values)
    mean = np.mean(values)
    b = n_std
    lower_limit = mean - b * std
    upper_limit = mean + b * std
    return lower_limit,upper_limit
