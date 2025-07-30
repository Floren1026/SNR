import os
import time
from utils.model_util import *

def file(args):
    ############  Create the dirs to save the result ############
    try:
        # cur_time = time.strftime('%Y-%b-%d_%H:%M', time.localtime())
        cur_time = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())
        if args.mode == 'train':
            secret_comment = 'color' if args.channel_secret == 3 else 'gray'
            cover_comment = 'color' if args.channel_cover == 3 else 'gray'
            comment = str(args.num_secret) + secret_comment + 'In' + str(args.num_cover) + cover_comment
            if args.train_continue:
                args.experiment_dir = args.Model_dir
            else:
                args.experiment_dir = cur_time + "_" + str(args.imageSize) + "_" + str(args.lr) + "_" + str(args.beta_R) + "_" + str(args.beta_hl) + "_" +str(args.beta_rl) + "_" + args.loss + "_" + comment + "_" + args.hostname
                # args.experiment_dir = cur_time + "_" + str(args.imageSize) + "_" + str(args.lr_H) + "_" + str(args.lr_R) + "_" + str(args.batch_stegs) + "_" + args.loss + "_"+ comment + "_" + args.hostname
            
            args.outckpts += args.experiment_dir + "/checkPoints"
            args.trainpics += args.experiment_dir + "/trainPics"
            args.validationpics += args.experiment_dir + "/validationPics"
            args.outlogs += args.experiment_dir + "/trainingLogs"
            args.outcodes += args.experiment_dir + "/codes"
            if not os.path.exists(args.outckpts):
                os.makedirs(args.outckpts)
            if not os.path.exists(args.trainpics):
                os.makedirs(args.trainpics)
            if not os.path.exists(args.validationpics):
                os.makedirs(args.validationpics)
            if not os.path.exists(args.outlogs):
                os.makedirs(args.outlogs)
            if not os.path.exists(args.outcodes):
                os.makedirs(args.outcodes)

            if not args.train_continue:
                save_current_codes(args.outcodes)
        else:
            args.experiment_dir = args.Model_dir
            args.testPics += args.experiment_dir + "/testPics"
            args.validationpics = args.testPics
            args.outlogs += args.experiment_dir + "/testLogs"
            if (not os.path.exists(args.testPics)) and args.Model_dir != '':
                os.makedirs(args.testPics)
            if not os.path.exists(args.outlogs):
                os.makedirs(args.outlogs)
    except OSError:
        print("mkdir failed   XXXXXXXXXXXXXXXXXXXXX") # ignore

    return args

def result_analysis(args):
    args.result_file = 'result_analysis'
    args.result_image = args.result_file + '/result'
    args.cover = args.result_image + '/cover'
    args.secret = args.result_image + '/secret/1'
    args.steg = args.result_image + '/steg/1'
    args.rev = args.result_image + '/rev'
    args.res_steg = args.result_image + '/res_steg'
    args.res_rev = args.result_image + '/res_rev'
    args.rev_from_local = args.result_image + '/rev_from_local'

    if not os.path.exists(args.result_file):
        os.makedirs(args.result_file)
    if not os.path.exists(args.result_image):
        os.makedirs(args.result_image)
    if not os.path.exists(args.cover):
        os.makedirs(args.cover)
    if not os.path.exists(args.secret):
        os.makedirs(args.secret)
    if not os.path.exists(args.steg):
        os.makedirs(args.steg)
    if not os.path.exists(args.rev):
        os.makedirs(args.rev)
    if not os.path.exists(args.res_steg):
        os.makedirs(args.res_steg)
    if not os.path.exists(args.res_rev):
        os.makedirs(args.res_rev)
    if not os.path.exists(args.rev_from_local):
        os.makedirs(args.rev_from_local)

    return args