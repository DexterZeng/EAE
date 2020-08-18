from include.Config import Config
import tensorflow as tf
from include.Model import build_SE, build_AE, training
from include.Test import get_combine_hits , get_hits_mrr, get_combine_hits_mrr
import time
from include.Load import *
import json
import scipy
from scipy import spatial
import copy
import numpy as np

seed = 12306
np.random.seed(seed)
tf.set_random_seed(seed)

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def make_print_to_file(fileName, path='./'):
    '''
    path， it is a path for save your log about fuction print
    example:
    use  make_print_to_file()   and the   all the information of funtion print , will be write in to a log file
    :return:
    '''
    import sys
    import os
    import sys
    import datetime

    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8',)

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    #fileName = datetime.datetime.now().strftime('day'+'%Y_%m_%d')
    sys.stdout = Logger(fileName + '.log', path=path)

    #############################################################
    # 这里输出之后的所有的输出的print 内容即将写入日志
    #############################################################
    print(fileName.center(60,'*'))

def getsim_matrix_cosine(se_vec, ne_vec, test_pair, method):
    Lvec = tf.placeholder(tf.float32, [None, se_vec.shape[1]])
    Rvec = tf.placeholder(tf.float32, [None, se_vec.shape[1]])
    he = tf.nn.l2_normalize(Lvec, dim=-1) #??? 规范化啊
    norm_e_em = tf.nn.l2_normalize(Rvec, dim=-1)
    aep = tf.matmul(he, tf.transpose(norm_e_em))

    Lvec_ne = tf.placeholder(tf.float32, [None, ne_vec.shape[1]])
    Rvec_ne = tf.placeholder(tf.float32, [None, ne_vec.shape[1]])
    he_n = tf.nn.l2_normalize(Lvec_ne, dim=-1) #??? 规范化啊
    norm_e_em_n = tf.nn.l2_normalize(Rvec_ne, dim=-1)
    aep_n = tf.matmul(he_n, tf.transpose(norm_e_em_n))

    sess = tf.Session()
    Lv = np.array([se_vec[e1] for e1, e2 in test_pair])
    Lid_record = np.array([e1 for e1, e2 in test_pair])
    Rv = np.array([se_vec[e2] for e1, e2 in test_pair])
    Rid_record = np.array([e2 for e1, e2 in test_pair])

    Lv_ne = np.array([ne_vec[e1] for e1, e2 in test_pair])
    Rv_ne = np.array([ne_vec[e2] for e1, e2 in test_pair])
    aep = sess.run(aep, feed_dict = {Lvec: Lv, Rvec: Rv})
    aep_n = sess.run(aep_n, feed_dict = {Lvec_ne: Lv_ne, Rvec_ne: Rv_ne})
    aep = 1-aep
    aep_n = 1-aep_n

    return aep, aep_n

def getsim_matrix(se_vec, test_pair, method):
    Lvec = np.array([se_vec[e1] for e1, e2 in test_pair])
    Rvec = np.array([se_vec[e2] for e1, e2 in test_pair])
    aep = scipy.spatial.distance.cdist(Lvec, Rvec, metric=method)

    return aep

def get_hits(vec, test_pair, method, top_k=(1, 10, 50, 100)):
    Lvec = np.array([vec[e1] for e1, e2 in test_pair])
    Rvec = np.array([vec[e2] for e1, e2 in test_pair])
    sim = scipy.spatial.distance.cdist(Lvec, Rvec, metric=method)
    top_lr = [0] * len(top_k)
    mrr_sum_l = 0
    for i in range(Lvec.shape[0]):
        rank = sim[i, :].argsort()
        # print(sim[i, :])
        rank_index = np.where(rank == i)[0][0]
        mrr_sum_l = mrr_sum_l + 1.0 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
    print('For each left:')
    for i in range(len(top_lr)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100))
    print("MRR: " + str(mrr_sum_l / len(test_pair)))

def get_hits_ma(sim, test_pair, top_k=(1, 10)):
    top_lr = [0] * len(top_k)
    mrr_sum_l = 0
    for i in range(sim.shape[0]):
        rank = sim[i, :].argsort()
        # print(sim[i, :])
        rank_index = np.where(rank == i)[0][0]
        mrr_sum_l = mrr_sum_l + 1.0 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
    msg = 'Hits@1:%.3f, Hits@10:%.3f, MRR:%.3f\n' % (top_lr[0] / len(test_pair), top_lr[1] / len(test_pair), mrr_sum_l / len(test_pair))
    print(msg)

def cal_performance(ranks, top=10):
    m_r = sum(ranks) * 1.0 / len(ranks)
    h_10 = sum(ranks <= top) * 1.0 / len(ranks)
    mrr = (1. / ranks).sum() / len(ranks)
    return m_r, h_10, mrr

def get_combine_hits_select_correct1(vec, name_vec, str_sim, test_pair):
    aep = getsim_matrix(vec, test, method)
    # aep_n = np.load('./data/' + Config.language + '/name_mat_train-' + method + '.npy')
    # weight_stru = 0.3
    # weight_text = 0.3
    # weight_string = 0.3
    # aep_fuse = (aep * weight_stru + aep_n * weight_text + str_sim * weight_string)
    # print(aep_fuse)
    # get_hits_ma(aep_fuse, test)

    aep_fuse = 1-aep
    aep_fuse_r = aep_fuse.T
    probs = aep_fuse - aep_fuse[range(len(test_pair)), range(len(test_pair))].reshape(len(aep_fuse), 1)
    # only rank those who have correspondings... cause the set is distorted for those above max_correct
    ranks = (probs >= 0).sum(axis=1)
    truth =  np.where(ranks==1)
    truths = truth[0].tolist()
    ind = np.argmax(probs, axis= 1)
    #ind = np.append(ind, np.array(cannotmactch))
    maxes = np.max(probs, axis= 1)
    probs[range(len(probs)),np.argmax(probs, axis= 1)] = np.min(probs)
    maxes1 = np.max(probs, axis= 1)
    gap = maxes-maxes1
    MR, H10, MRR = cal_performance(ranks, top=10)
    _, H1, _ = cal_performance(ranks, top=1)

    msg = 'Hits@1:%.3f, Hits@10:%.3f, MR:%.3f, MRR:%.3f' % (H1, H10, MR, MRR)
    print('\n'+msg)

    # right
    probs = aep_fuse_r - aep_fuse_r[range(len(test_pair)), range(len(test_pair))].reshape(len(aep_fuse_r), 1)
    ranks_r = (probs >= 0).sum(axis=1)
    truth_r =  np.where(ranks_r==1)
    truths_r = truth_r[0].tolist()
    ind_r = np.argmax(probs, axis= 1)
    maxes = np.max(probs, axis= 1)
    probs[range(len(probs)),np.argmax(probs, axis= 1)] = np.min(probs)
    maxes1 = np.max(probs, axis= 1)
    gap_r = maxes-maxes1
    MR, H10, MRR = cal_performance(ranks_r, top=10)
    _, H1, _ = cal_performance(ranks_r, top=1)
    msg = 'Hits@1:%.3f, Hits@10:%.3f, MR:%.3f, MRR:%.3f' % (H1, H10, MR, MRR)
    print('\n'+msg)
    return ind, gap, truths, ranks,  ind_r, gap_r, truths_r, ranks_r

def rep_match(ori, index1, index2, gap1, gap2, ranks1):
	dicrank = dict() # only exist for one round!!! cause the train is also updating...
	coun = 0
	truecounter = 0
	addedtrain = []
	for i in range(len(index1)):
		if index1[i] < len(index2):
			if index2[index1[i]] == i:
				# if gap1[i] >= 0.03 and gap2[i] >= 0.03:
					coun += 1
					dicrank[str(test[i][0])] = ranks1[i] # records the ranks of confident results, should be 1
					# wrong... you just directly removed the right pair??? nonono... so complicated
					### might be wrong, but remove the wrongly selected one
					addedtrain.append(tuple([int(test[i][0]), int(test[index1[i]][1])])) # add the wrong one
					#newtest.remove(tuple([int(test[i][0]), int(test[i][1])]))
					if test[i][0] + 10500 == test[index1[i]][1]:
						truecounter += 1
	print(coun)
	print(truecounter)
	addedtrain.extend(ori)
	return addedtrain, dicrank

if __name__ == '__main__':
    make_print_to_file(Config.language, path='./logs/')
    t = time.time()
    e = len(set(loadfile(Config.e1, 1)) | set(loadfile(Config.e2, 1)))
    print(e)
    ILL = loadfile(Config.ill, 2)
    illL = len(ILL)
    # np.random.shuffle(ILL)
    # train = np.array(ILL[:illL // 10 * Config.seed])
    # train_ori = copy.deepcopy(train)
    # train_array = np.array(train)
    # test = ILL[illL // 10 * Config.seed:]
    test = ILL[:17880]
    train = np.array(ILL[17880:])
    train_ori = copy.deepcopy(train)
    train_array = np.array(train)
    KG1 = loadfile(Config.kg1, 3)
    KG2 = loadfile(Config.kg2, 3)

    storepath = './data/' + Config.language + '/se_ite/'
    np.save(storepath + 'train.npy', train_array)
    np.save(storepath + 'test.npy', test)
    ite_counter = 0

    # build SE
    # output_layer, loss,= build_SE(Config.se_dim, Config.act_func, Config.gamma, Config.k, e, train, KG1 + KG2)
    # se_vec, J = training(output_layer, loss, 25, Config.epochs_se, train, e, Config.k)
    # np.save('./data/' + Config.language + '/se_vec_test.npy', se_vec)
    # print('loss:', J)
    # print('Result of SE:')

    se_vec = np.load('./data/' + Config.language + '/se_vec_test.npy')
    # get_hits_mrr(se_vec, test)
    # get_hits(se_vec, test, 'braycurtis')
    if '_V1' in Config.language:
        # nepath = './data/' + Config.language + '/name_vec_cpm_3.txt'
        nepath = './data/' + Config.language + '/name_vec_ftext.txt'
        ne_vec = loadNe(nepath)
        # ne_vecold = copy.deepcopy(ne_vec)
        # print("process...")
        # print(ne_vec)
        # for i in range(len(ne_vec)):
        #     if sum(ne_vec[i]) == 0:
        #         print(ne_vec[i])
        #         print(i)
        #         ne_vec[i][0] = 0.000001
        # print("done...")
    else:
        with open(file='./data/' + Config.language + '/' + Config.language.split('_')[0] + '_vectorList.json',
                  mode='r', encoding='utf-8') as f:
            embedding_list = json.load(f)
            print(len(embedding_list), 'rows,', len(embedding_list[0]), 'columns.')
            ne_vec = np.array(embedding_list)
    str_sim = np.load('./data/' + Config.language + '/string_mat_train.npy')
    str_sim = 1 - str_sim
    # print(str_sim)
    # get_hits_ma(str_sim, test)

    method = 'euclidean'  # cosine,cityblock euclidean

    index1, gap1, truths1, ranks1, index2, gap2, truths2, ranks2 = get_combine_hits_select_correct1(se_vec, ne_vec,str_sim,
                                                                                                    test)

    trainlength_old = len(train)
    train, dicrank = rep_match(train_ori, index1, index2, gap1, gap2, ranks1)
    train_array = np.array(train)  # array

    print('len of new train/seed: ' + str(len(train)))
    np.save(storepath + 'train.npy', train)

    while len(train) - trainlength_old >= 20:
        ite_counter += 1
        output_layer, loss = build_SE(Config.se_dim, Config.act_func, Config.gamma, Config.k, e, train_array, KG1 + KG2)
        se_vec, J = training(output_layer, loss, 25, Config.epochs_se, train_array, e, Config.k)
        np.save(storepath + 'se_vec' + str(ite_counter) + '.npy', se_vec)
        print('loss:', J)
        print('Result of SE:')
        index1, gap1, truths1, ranks1, index2, gap2, truths2, ranks2 = get_combine_hits_select_correct1(se_vec, ne_vec,str_sim,
                                                                                                        test)
        trainlength_old = len(train)
        train, dicrank = rep_match(train_ori, index1, index2, gap1, gap2, ranks1)
        train_array = np.array(train)  # array
        print('len of new train/seed: ' + str(len(train)))
        np.save(storepath + 'train' + str(ite_counter) + '.npy', train)

    # get_hits_mrr(se_vec, test)
    # if '_V1' in Config.language:
    #     aep_nnew = copy.deepcopy(aep_n)
    #     counter = 0
    #     totoal = 0
    #     for e1, e2 in vali:
    #         if sum(ne_vecold[e1]) == 0:
    #             #print(e1)
    #             totoal +=1
    #             aep_nnew[counter] = np.ones(10500)
    #         if sum(ne_vecold[e2]) == 0:
    #             aep_nnew[:, counter] = np.ones(10500)
    #         counter += 1
    #     print(totoal)
    #     np.save('./data/' + Config.language + '/name_mat_train_new-' + method + '.npy', aep_nnew)
    #     get_hits_ma(aep_nnew, vali)
    #     aep_fuse = (aep * weight_stru + aep_nnew * weight_text + str_sim * weight_string)
    #     print(aep_fuse)
    #     get_hits_ma(aep_fuse, vali)

    #index1, gap1, truths1, ranks1, index2, gap2, truths2, ranks2 = get_hits_select_correct_1(se_vec, test)


    # addnewents
