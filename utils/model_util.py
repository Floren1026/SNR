import os
import csv
import cv2
import torch
import torch.optim
import shutil
import torchvision.utils as vutils


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# custom weights initialization called on net
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def print_log(log_info, log_path):
    print(log_info)
    if not os.path.exists(log_path):
        fp = open(log_path, "w")
        fp.writelines(log_info + "\n")
    else:
        with open(log_path, 'a+') as f:
            f.writelines(log_info + '\n')


# Print the structure and parameters number of the net
def print_network(net, logPath):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print_log(str(net), logPath)
    print_log('Total number of parameters: %d' % num_params, logPath)


# Code saving
def save_current_codes(des_path):
    main_file_path = os.path.realpath(__file__)
    cur_work_dir = os.path.dirname(os.path.split(main_file_path)[0])

    data_dir = cur_work_dir + "/data/"
    new_data_dir_path = des_path + "/data/"
    shutil.copytree(data_dir, new_data_dir_path)

    model_dir = cur_work_dir + "/models/"
    new_model_dir_path = des_path + "/models/"
    shutil.copytree(model_dir, new_model_dir_path)

    utils_dir = cur_work_dir + "/utils/"
    new_utils_dir_path = des_path + "/utils/"
    shutil.copytree(utils_dir, new_utils_dir_path)

    train_file = cur_work_dir + '/train.py'
    new_train_file_path = des_path + '/train.py'
    shutil.copyfile(train_file, new_train_file_path)

    test_file = cur_work_dir + '/test.py'
    new_test_file_path = des_path + '/test.py'
    shutil.copyfile(test_file, new_test_file_path)


def save_networks(args, state, is_best, epoch, prefix, mode):

    checkpoint_epoch='%s/checkpoint_%.3i_%s.pt'% (args.outckpts, epoch, mode)
    checkpoint='%s/checkpoint_%s.pt'% (args.outckpts, mode)
    torch.save(state, checkpoint)

    if epoch % args.save_freq == 0:
        torch.save(state, checkpoint_epoch)
    if is_best:
        shutil.copyfile(checkpoint, '%s/best_checkpoint_%s.pt'% (args.outckpts, mode))
    if epoch == args.epochs-1:
        with open(prefix + '.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([state])


def gauss_noise(shape):
    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()

    return noise


def save_model(args, state, is_best, epoch, prefix):

    checkpoint_epoch='%s/checkpoint_%.3i.pt'% (args.outckpts, (epoch+1))
    checkpoint='%s/checkpoint.pt'% (args.outckpts)
    torch.save(state, checkpoint)

    if (epoch+1) % args.save_freq == 0:
        torch.save(state, checkpoint_epoch)
    if is_best:
        shutil.copyfile(checkpoint, '%s/best_checkpoint.pt'% (args.outckpts))
    if epoch == args.epochs-1:
        with open(prefix + '.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([state])


def save_result_pic(args, cover_img, steg_img, secret_img, rev_img, epoch, i, save_path):

    resultImgName = '%s/ResultPics_epoch%03d_batch%04d.png' % (save_path, epoch, i)

    if args.cuda:
        cover_img = cover_img.cuda()
        steg_img = steg_img.cuda()
        rev_img = rev_img.cuda()
        secret_img = secret_img.cuda()

    cover_gap = steg_img - cover_img
    secret_gap = rev_img - secret_img
    cover_gap = (cover_gap*10 + 0.5).clamp_(0.0, 1.0)
    secret_gap = (secret_gap*10 + 0.5).clamp_(0.0, 1.0)

    for i in range(args.num_cover):
        cover_i = cover_img[:,i*args.channel_cover:(i+1)*args.channel_cover,:,:]
        steg_i = steg_img[:,i*args.channel_cover:(i+1)*args.channel_cover,:,:]
        cover_gap_i = cover_gap[:,i*args.channel_cover:(i+1)*args.channel_cover,:,:]

        if i == 0:
            showCover = torch.cat((cover_i, steg_i, cover_gap_i),0)
        else:
            showCover = torch.cat((showCover, cover_i, steg_i, cover_gap_i),0)

    for i_secret in range(args.num_secret):
        secret_i = secret_img[:,i_secret*args.channel_secret:(i_secret+1)*args.channel_secret,:,:]
        rev_secret_i = rev_img[:,i_secret*args.channel_secret:(i_secret+1)*args.channel_secret,:,:]
        secret_gap_i = secret_gap[:,i_secret*args.channel_secret:(i_secret+1)*args.channel_secret,:,:]

        if i_secret == 0:
            showSecret = torch.cat((secret_i, rev_secret_i, secret_gap_i),0)
        else:
            showSecret = torch.cat((showSecret, secret_i, rev_secret_i, secret_gap_i),0)


    if args.channel_secret == args.channel_cover:
        showAll = torch.cat((showCover, showSecret),0)
        vutils.save_image(showAll, resultImgName, nrow=2, padding=1, normalize=True)
    else:
        ContainerImgName = '%s/ContainerPics_epoch%03d_batch%04d.png' % (save_path, epoch, i)
        SecretImgName = '%s/SecretPics_epoch%03d_batch%04d.png' % (save_path, epoch, i)
        vutils.save_image(showCover, ContainerImgName, nrow=3*(args.num_cover+args.num_secret), padding=1, normalize=True)
        vutils.save_image(showSecret, SecretImgName, nrow=3*(args.num_cover+args.num_secret), padding=1, normalize=True)
