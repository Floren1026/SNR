import os
import time
import socket
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from utils.dirs2save import *
from utils.model_util import *
from data.datasets import Mydata
from utils.calculate_PSNR_SSIM import *
from models.wavelet import DWT_2D, IDWT_2D
from models.SNR import UISTransformer as UISTnet


parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', help='train or test')
parser.add_argument('--num_workers', type=int, default=2,
                    help='number of data loading workers')
parser.add_argument('--imageSize', type=int, default=256,
                    help='the size of image')
parser.add_argument('--net_imagesize', type=int, default=128,
                    help='the number of frames')
parser.add_argument('--norm', default='instance', help='batch or instance')
parser.add_argument('--loss', default='l2', help='l1 or l2')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train for')
parser.add_argument('--iters_epochs', type=int, default=20000, 
                    help='train numbers of each epoch')
parser.add_argument('--lr', type=float, default=0.0003,
                    help='Hnet learning rate, default=0.0001')
parser.add_argument('--beta_R', type=float, default=2, help='hyper parameter beta of reveal')
parser.add_argument('--beta_hl', type=float, default=0.75, help='hyper parameter beta of the low-frequency hiding')
parser.add_argument('--beta_rl', type=float, default=0.75, help='hyper parameter beta of the low-frequency revealing')
parser.add_argument('--hostname', default=socket.gethostname(), 
                    help='the host name of the running server')
parser.add_argument('--cuda', type=bool, default=True,
                    help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of GPUs to use')
parser.add_argument('--trainpics', default='./training/',
                    help='folder to output training images')
parser.add_argument('--validationpics', default='./training/',
                    help='folder to output validation images')
parser.add_argument('--testPics', default='./training/',
                    help='folder to output test images')
parser.add_argument('--outckpts', default='./training/',
                    help='folder to output checkpoints')
parser.add_argument('--outlogs', default='./training/',
                    help='folder to output images')
parser.add_argument('--outcodes', default='./training/',
                    help='folder to save the experiment codes')
# whether to continue training
parser.add_argument('--train_continue', type=bool, default=False, 
                    help='continue to training')
parser.add_argument('--trained_epochs', type=int, default=0, help='the epochs of the train')
parser.add_argument('--Model_dir', default='', help='the dir of model')
parser.add_argument('--model', default='checkpoint.pt', help='the filename of the checkpoint')
parser.add_argument('--save_freq', default=10, help='How many epochs to save model')
parser.add_argument('--logFrequency', type=int, default=10, help='the frequency of print the log on the console')
parser.add_argument('--resultPicFrequency', type=int, default=100, help='the frequency of save the resultPic')
parser.add_argument('--num_secret', type=int, default=1, help='How many secret images are hidden in one cover image?')
parser.add_argument('--num_cover', type=int, default=1, help='How many secret images are hidden in one cover image?')
parser.add_argument('--batch_stegs', type=int, default=1, help='How many stegs for one batch')
parser.add_argument('--channel_cover', type=int, default=3, help='1: gray; 3: color')
parser.add_argument('--channel_secret', type=int, default=3, help='1: gray; 3: color')




def main():
    ############### define global parameters ###############
    global args, optimizer, writer, logPath, scheduler, valLoader, smallestLoss

    args = parser.parse_args()
    args.ngpu = torch.cuda.device_count()
    if torch.cuda.is_available() and not args.cuda:
        print("You are not using GPU, it is recommended to use CUDA to run")

    cudnn.benchmark = True

    args = file(args)

    logPath = args.outlogs + '/log.txt'
    print_log(str(args), logPath)

    ##################  Model initialize  ##################
    Net = UISTnet(img_size=args.net_imagesize, in_chans=args.num_secret*args.channel_secret+args.num_cover*args.channel_cover, 
               out_chans=args.num_secret*args.channel_secret, window_size=8, img_range=1., depths=[6, 6], embed_dim=36, 
               num_heads=[6, 6], mlp_ratio=2)
    
    Net.cuda()

    if args.ngpu > 1:
        Net = nn.DataParallel(Net).cuda()

    # load model
    if args.train_continue:
        checkpoint = "./training/" + args.Model_dir + "/checkPoints/"
        checkpointH = torch.load(checkpoint + args.model)
        Net.load_state_dict(checkpointH['state_dict'])

    print_network(Net, logPath)


    # Loss and Metric
    if args.loss == 'l1':
        criterion = nn.L1Loss().cuda()
    if args.loss == 'l2':
        criterion = nn.MSELoss().cuda()

    optimizer = optim.Adam(Net.parameters(), lr=args.lr, weight_decay=0) 
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20, 30, 40, 50], 0.5)

    # write the networks
    writer = SummaryWriter(log_dir='runs/' + args.experiment_dir)


    ##################  Datasets  ##################
    DATA_DIR = '/home/user-yc/imagenet'

    traindir = os.path.join(DATA_DIR, 'train')
    valdir = os.path.join(DATA_DIR, 'val')

    batch_size_C=args.batch_stegs*args.num_cover
    batch_size_S=args.batch_stegs*args.num_secret


    train_loader_cover = Mydata(args, traindir, batch_size_C)
    train_loader_secret = Mydata(args, traindir, batch_size_S)

    val_loader_cover = Mydata(args, valdir, batch_size_C)
    val_loader_secret = Mydata(args, valdir, batch_size_S)


    ##################  start to training  ##################
    print_log("............................ start to training ...........................", logPath)
    smallestLoss = 10000
    for epoch in range(args.epochs):
        trainLoader = zip(train_loader_cover, train_loader_secret)
        valLoader = zip(val_loader_cover, val_loader_secret)
        epoch = epoch + args.trained_epochs
        
        train(trainLoader, epoch, Net=Net, criterion=criterion)

        val_hloss, val_rloss, val_hlow_loss, val_rlow_loss, val_hdiff, val_rdiff = validation(valLoader, epoch, Net=Net, criterion=criterion)

        total_loss = val_hloss + args.beta_R * val_rloss +  args.beta_hl * val_hlow_loss + args.beta_rl * val_rlow_loss
        scheduler.step()

        # save the best model parameters
        is_best = total_loss < globals()["smallestLoss"]
        globals()["smallestLoss"] = total_loss

        save_model(args, {
            'epoch': epoch,
            'state_dict': Net.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, is_best, epoch, '%s/epoch_%d_Hloss_%.4f_Hdiff=%.4f_Rloss_%.4f_Rdiff=%.4f_Total_loss_%.4f' 
        % (args.outckpts, epoch, val_hloss, val_hdiff, val_rloss, val_rdiff, total_loss))

        if epoch == args.epochs:
            break

    writer.close()


def train(trainLoader, epoch, Net, criterion):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    Hlosses = AverageMeter()
    Rlosses = AverageMeter()
    H_low_losses = AverageMeter()
    R_low_losses = AverageMeter()
    TotalLosses = AverageMeter()  # Hloss + β*Rloss + hl*hiding_low + rl*revealing_low
    Hdiff = AverageMeter()
    Rdiff = AverageMeter()
    # PSNR_C = AverageMeter()
    # PSNR_S = AverageMeter()
    # SSIM_C = AverageMeter()
    # SSIM_S = AverageMeter()

    # switch to train mode
    Net.train()

    start_time = time.time()
    # for i, data in enumerate(trainLoader, 0):
    for i, ((cover_img, cover_target), (secret_img, secret_target)) in enumerate(trainLoader, 0):

        data_time.update(time.time() - start_time)

        cover_imgv, steg_img, secret_imgv_r, rev_img, errH, errR, H_low_err, R_low_err, diffH, diffR = steg(args, cover_img, secret_img, Net, criterion)

        # Loss function
        err_total = errH + args.beta_R * errR + args.beta_hl * H_low_err + args.beta_rl * R_low_err

        Hlosses.update(errH.item(), args.batch_stegs*args.num_cover)
        Rlosses.update(errR.item(), args.batch_stegs*args.num_secret)
        Hdiff.update(diffH.item(), args.batch_stegs*args.num_cover)
        Rdiff.update(diffR.item(), args.batch_stegs*args.num_secret)
        H_low_losses.update(H_low_err.item(), args.batch_stegs*args.num_cover)
        R_low_losses.update(R_low_err.item(), args.batch_stegs*args.num_secret)
        TotalLosses.update(err_total.item(), args.batch_stegs*(args.num_cover+args.num_secret))
        # PSNR_C.update(psnr_c)
        # PSNR_S.update(psnr_s)
        # SSIM_C.update(ssim_c)
        # SSIM_S.update(ssim_s)
        
        optimizer.zero_grad()
        err_total.backward()
        optimizer.step()

        # Time spents on one batch
        batch_time.update(time.time() - start_time)
        start_time = time.time()

        log = '[%d/%d][%d/%d]\tLoss_H: %.6f Loss_R: %.6f L1_H_diff: %.4f L1_R_diff: %.4f Loss_sum: %.6f \tdatatime: %.6f \tbatchtime: %.6f' % (
            epoch, args.epochs, i, args.iters_epochs,
            Hlosses.val, Rlosses.val, Hdiff.val, Rdiff.val, TotalLosses.val, data_time.val, batch_time.val)

        if i % args.logFrequency == 0:
            print(log)

        # genereate a picture every resultPicFrequency steps
        if epoch % 1 == 0 and i % args.resultPicFrequency == 0:
            save_result_pic(args, cover_imgv, steg_img.data, secret_imgv_r, rev_img.data, epoch, i, args.trainpics)
            
        if i == args.iters_epochs-1:
            break

    train_log = "Training[%d] Hloss= %.6f\tRloss= %.6f\tHdiff= %.4f\tRdiff= %.4f\tlr= %.6f\t Epoch time= %.4f" % (
        epoch, Hlosses.avg, Rlosses.avg, Hdiff.avg, Rdiff.avg, optimizer.param_groups[0]['lr'], batch_time.sum)
    
    print_log(train_log, logPath)


    writer.add_scalar("lr/lr", optimizer.param_groups[0]['lr'], epoch)
    writer.add_scalar('train/R_loss', Rlosses.avg, epoch)
    writer.add_scalar('train/H_loss', Hlosses.avg, epoch)
    writer.add_scalar('train/H_low_loss', H_low_losses.avg, epoch)
    writer.add_scalar('train/R_low_loss', R_low_losses.avg, epoch)
    writer.add_scalar('train/total_loss', TotalLosses.avg, epoch)    
    writer.add_scalar('train/H_diff', Hdiff.avg, epoch)
    writer.add_scalar('train/R_diff', Rdiff.avg, epoch)
    # writer.add_scalar('train/psnr_c', PSNR_C.avg, epoch)
    # writer.add_scalar('train/ssim_c', SSIM_C.avg, epoch)
    # writer.add_scalar('train/psnr_s', PSNR_S.avg, epoch)
    # writer.add_scalar('train/ssim_s', SSIM_S.avg, epoch)

    return Hdiff.avg, Rdiff.avg, TotalLosses.avg


def validation(valLoader, epoch, Net, criterion):
    print("#################################################### validation begin ########################################################")

    Hlosses = AverageMeter()
    Rlosses = AverageMeter()
    H_low_losses = AverageMeter()
    R_low_losses = AverageMeter()
    TotalLosses = AverageMeter()
    Hdiff = AverageMeter()
    Rdiff = AverageMeter()
    # PSNR_C = AverageMeter()
    # PSNR_S = AverageMeter()
    # SSIM_C = AverageMeter()
    # SSIM_S = AverageMeter()


    start_time = time.time()

    with torch.no_grad():
        Net.eval()

        for i, ((cover_img, cover_target), (secret_img, secret_target)) in enumerate(valLoader, 0):

            cover_imgv, steg_img, secret_imgv_r, rev_img, errH, errR, H_low_err, R_low_err, diffH, diffR = steg(args, cover_img, secret_img, Net, criterion)

            # Loss function
            err_total = errH + args.beta_R * errR + args.beta_hl * H_low_err + args.beta_rl * R_low_err

            Hlosses.update(errH.item(), args.batch_stegs*args.num_cover)
            Rlosses.update(errR.item(), args.batch_stegs*args.num_secret)
            H_low_losses.update(H_low_err.item(), args.batch_stegs*args.num_cover)
            R_low_losses.update(R_low_err.item(), args.batch_stegs*args.num_secret)
            Hdiff.update(diffH.item(), args.batch_stegs*args.num_cover)
            Rdiff.update(diffR.item(), args.batch_stegs*args.num_secret)  
            TotalLosses.update(err_total.item(), args.batch_stegs*(args.num_cover+args.num_secret))
            # PSNR_C.update(psnr_c)
            # PSNR_S.update(psnr_s)
            # SSIM_C.update(ssim_c)
            # SSIM_S.update(ssim_s)

            if i == 0:
                save_result_pic(args, cover_imgv, steg_img.data, secret_imgv_r, rev_img.data, epoch, i, args.validationpics)

            if i == 200-1:
                break    

    val_time = time.time() - start_time
    
    val_log = "validation[%d] val_Hloss = %.6f\t val_Rloss = %.6f\t val_Hdiff = %.4f\t val_Rdiff=%.4f\t validation time=%.2f" % (
        epoch, Hlosses.avg, Rlosses.avg, Hdiff.avg, Rdiff.avg, val_time)
    print_log(val_log, logPath)


    writer.add_scalar('validation/H_loss', Hlosses.avg, epoch)
    writer.add_scalar('validation/R_loss', Rlosses.avg, epoch)
    writer.add_scalar('validation/H_low_loss', H_low_losses.avg, epoch)
    writer.add_scalar('validation/R_low_loss', R_low_losses.avg, epoch)
    writer.add_scalar('validation/total_loss', TotalLosses.avg, epoch) 
    writer.add_scalar('validation/H_diff', Hdiff.avg, epoch)
    writer.add_scalar('validation/R_diff', Rdiff.avg, epoch)
    # writer.add_scalar('validation/psnr_c', PSNR_C.avg, epoch)
    # writer.add_scalar('validation/ssim_c', SSIM_C.avg, epoch)
    # writer.add_scalar('validation/psnr_s', PSNR_S.avg, epoch)
    # writer.add_scalar('validation/ssim_s', SSIM_S.avg, epoch)

    print(
        "#################################################### validation end ########################################################")
    return Hlosses.avg, Rlosses.avg, H_low_losses.avg, R_low_losses.avg, Hdiff.avg, Rdiff.avg


def steg(args, cover_img, secret_img, Net, criterion):

    dwt = DWT_2D(wave='haar')

    batch_size_cover, channel_cover, _, _ = cover_img.size()
    batch_size_secret, channel_secret, _, _ = secret_img.size()

    if args.cuda:
        cover_img = cover_img.cuda()
        secret_img = secret_img.cuda()

    # Adjust the tensor shape based on the number of cover and secret images
    cover_imgv = cover_img.view(batch_size_cover // args.num_cover, 
                                channel_cover * args.num_cover, args.imageSize, args.imageSize)
    secret_imgv = secret_img.view(batch_size_secret // args.num_secret, 
                                channel_secret * args.num_secret, args.imageSize, args.imageSize)        
    secret_imgv_r = secret_imgv.repeat(1,1,1,1)

    concat_img = torch.cat((cover_imgv, secret_imgv), dim=1)
    # hiding
    steg_img = Net(concat_img)

    steg_guass = gauss_noise(steg_img.shape)
    rev_cat = torch.cat((steg_img, steg_guass), 1)
    #revealing
    rev_img = Net(rev_cat)


    # loss between cover/steg and secret/reveal image 
    errH = criterion(steg_img, cover_imgv)   
    errR = criterion(rev_img, secret_imgv_r)

    cover_dwt = dwt(cover_imgv)
    secret_dwt = dwt(secret_imgv)
    steg_dwt = dwt(steg_img)
    rev_dwt = dwt(rev_img)
    # loss between the low-frequency of steg and cover image
    steg_low_loss = steg_dwt.narrow(1, 0, 3)
    cover_low_loss = cover_dwt.narrow(1, 0, 3)
    hiding_low_err = criterion(steg_low_loss, cover_low_loss)

    # loss between the low-frequency of reveal and secret image
    rev_low_loss =  rev_dwt.narrow(1, 0, 3)
    secret_low_loss = secret_dwt.narrow(1, 0, 3)
    revealing_low_err = criterion(rev_low_loss, secret_low_loss)


    # L1 metric
    diffH = (steg_img - cover_imgv).abs().mean()*255
    diffR = (rev_img - secret_imgv_r).abs().mean()*255

    # apd, ssim, psnr_c, ssim_c = tensor_psnr_ssim(cover_imgv, steg_img)
    # apd, ssim, psnr_s, ssim_s = tensor_psnr_ssim(secret_imgv_r, rev_img)

    return cover_imgv, steg_img, secret_imgv_r, rev_img, errH, errR, hiding_low_err, \
        revealing_low_err, diffH, diffR
  # \, psnr_c, ssim_c, psnr_s, ssim_s


if __name__ == '__main__':
    main()
