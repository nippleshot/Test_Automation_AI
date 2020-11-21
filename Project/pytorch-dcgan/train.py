from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *
from dataset import *
from util import *

import matplotlib.pyplot as plt

from torchvision import transforms

def train(args):
    ## 设置hyperparameter
    mode = args.mode
    train_continue = args.train_continue

    lr = args.lr
    batch_size = args.batch_size
    num_epoch = args.num_epoch

    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir
    log_dir = args.log_dir
    result_dir = args.result_dir

    task = args.task
    opts = [args.opts[0], np.asarray(args.opts[1:]).astype(np.float)]

    ny = args.ny
    nx = args.nx
    nch = args.nch
    nker = args.nker

    network = args.network
    learning_type = args.learning_type

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("mode: %s" % mode)

    print("learning rate: %.4e" % lr)
    print("batch size: %d" % batch_size)
    print("number of epoch: %d" % num_epoch)

    print("task: %s" % task)
    print("opts: %s" % opts)

    print("network: %s" % network)
    print("learning type: %s" % learning_type)

    print("data dir: %s" % data_dir)
    print("ckpt dir: %s" % ckpt_dir)
    print("log dir: %s" % log_dir)
    print("result dir: %s" % result_dir)

    print("device: %s" % device)

    ## 生成directory
    result_dir_train = os.path.join(result_dir, 'train')

    if not os.path.exists(result_dir_train):
        os.makedirs(os.path.join(result_dir_train, 'png'))

    ## 准备图像数据
    if mode == 'train':
        transform_train = transforms.Compose([Resize(shape=(ny, nx, nch)), Normalization(mean=0.5, std=0.5)])

        dataset_train = Dataset(data_dir=data_dir, transform=transform_train, task=task, opts=opts)

        loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)

        num_data_train = len(dataset_train)
        num_batch_train = np.ceil(num_data_train / batch_size)

    ## 生成DCGAN网络
    if network == "DCGAN":
        netG = DCGAN(in_channels=100, out_channels=nch, nker=nker).to(device)
        netD = Discriminator(in_channels=nch, out_channels=1, nker=nker).to(device)

        init_weights(netG, init_type='normal', init_gain=0.02)
        init_weights(netD, init_type='normal', init_gain=0.02)

    ## 定义loss函数
    fn_loss = nn.BCELoss().to(device)

    ## 设置Optimizer
    optimG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    optimD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

    ## 写Tensorboard的log的时候，使用的函数
    fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    fn_denorm = lambda x, mean, std: (x * std) + mean
    cmap = None

    ## 为了使用Tensorboard，设置SummaryWriter
    writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))


    ## 训练DCGAN
    st_epoch = 0
    if mode == 'train':
        if train_continue == "on":
            netG, netD, optimG, optimD, st_epoch = load(ckpt_dir=ckpt_dir,
                                                        netG=netG, netD=netD,
                                                        optimG=optimG, optimD=optimD)

        for epoch in range(st_epoch + 1, num_epoch + 1):
            netG.train()
            netD.train()

            loss_G_train = []
            loss_D_real_train = []
            loss_D_fake_train = []

            for batch, data in enumerate(loader_train, 1):
                # forward pass
                label = data['label'].to(device)
                input = torch.randn(label.shape[0], 100, 1, 1,).to(device)

                output = netG(input)

                # backward netD(判别器网络)
                set_requires_grad(netD, True)
                optimD.zero_grad()

                pred_real = netD(label)
                pred_fake = netD(output.detach())

                loss_D_real = fn_loss(pred_real, torch.ones_like(pred_real))
                loss_D_fake = fn_loss(pred_fake, torch.zeros_like(pred_fake))
                loss_D = 0.5 * (loss_D_real + loss_D_fake)

                loss_D.backward()
                optimD.step()

                # backward netG(生成器网络)
                set_requires_grad(netD, False)
                optimG.zero_grad()

                pred_fake = netD(output)

                loss_G = fn_loss(pred_fake, torch.ones_like(pred_fake))

                loss_G.backward()
                optimG.step()

                # 计算loss函数
                loss_G_train += [loss_G.item()]
                loss_D_real_train += [loss_D_real.item()]
                loss_D_fake_train += [loss_D_fake.item()]

                print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | "
                      "GEN %.4f | DISC REAL: %.4f | DISC FAKE: %.4f" %
                      (epoch, num_epoch, batch, num_batch_train,
                       np.mean(loss_G_train), np.mean(loss_D_real_train), np.mean(loss_D_fake_train)))

                if batch % 20 == 0:
                  # 保存到Tensorboard
                  output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5)).squeeze()
                  output = np.clip(output, a_min=0, a_max=1)

                  id = num_batch_train * (epoch - 1) + batch

                  plt.imsave(os.path.join(result_dir_train, 'png', '%04d_output.png' % id), output[0].squeeze(), cmap=cmap)
                  writer_train.add_image('output', output, id, dataformats='NHWC')

            writer_train.add_scalar('loss_G', np.mean(loss_G_train), epoch)
            writer_train.add_scalar('loss_D_real', np.mean(loss_D_real_train), epoch)
            writer_train.add_scalar('loss_D_fake', np.mean(loss_D_fake_train), epoch)

            if epoch % 2 == 0 or epoch == num_epoch:
                save(ckpt_dir=ckpt_dir, netG=netG, netD=netD, optimG=optimG, optimD=optimD, epoch=epoch)

        writer_train.close()

def test(args):
    ## 设置hyperparameter
    mode = args.mode
    lr = args.lr
    batch_size = args.batch_size
    num_epoch = args.num_epoch

    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir
    log_dir = args.log_dir
    result_dir = args.result_dir

    task = args.task
    opts = [args.opts[0], np.asarray(args.opts[1:]).astype(np.float)]

    ny = args.ny
    nx = args.nx
    nch = args.nch
    nker = args.nker

    network = args.network
    learning_type = args.learning_type

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("mode: %s" % mode)

    print("learning rate: %.4e" % lr)
    print("batch size: %d" % batch_size)
    print("number of epoch: %d" % num_epoch)

    print("task: %s" % task)
    print("opts: %s" % opts)

    print("network: %s" % network)
    print("learning type: %s" % learning_type)

    print("data dir: %s" % data_dir)
    print("ckpt dir: %s" % ckpt_dir)
    print("log dir: %s" % log_dir)
    print("result dir: %s" % result_dir)

    print("device: %s" % device)

    ##生成 directory
    result_dir_test = os.path.join(result_dir, 'test')

    if not os.path.exists(result_dir_test):
        os.makedirs(os.path.join(result_dir_test, 'png'))
        os.makedirs(os.path.join(result_dir_test, 'numpy'))

    ## 准备数据
    if mode == "test":
        transform_test = transforms.Compose([Resize(shape=(ny, nx, nch)), Normalization(mean=0.5, std=0.5)])

        dataset_test = Dataset(data_dir=data_dir, transform=transform_test, task=task, opts=opts)
        loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)

        num_data_test = len(dataset_test)
        num_batch_test = np.ceil(num_data_test / batch_size)

    ## 生成网络
    if network == "DCGAN":
        netG = DCGAN(in_channels=100, out_channels=nch, nker=nker).to(device)
        netD = Discriminator(in_channels=nch, out_channels=1, nker=nker).to(device)

        init_weights(netG, init_type='normal', init_gain=0.02)
        init_weights(netD, init_type='normal', init_gain=0.02)

    ## 设置Optimizer
    optimG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    optimD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

    fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    fn_denorm = lambda x, mean, std: (x * std) + mean
    cmap = None

    ## 生成新的测试图像数据
    if mode == "test":
        netG, netD, optimG, optimD, st_epoch = load(ckpt_dir=ckpt_dir, netG=netG, netD=netD, optimG=optimG, optimD=optimD)

        with torch.no_grad():
            netG.eval()

            input = torch.randn(batch_size, 100, 1, 1).to(device)
            output = netG(input)

            output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5))

            for j in range(output.shape[0]):
                id = j

                output_ = output[j]
                np.save(os.path.join(result_dir_test, 'numpy', '%04d_output.npy' % id), output_)

                output_ = np.clip(output_, a_min=0, a_max=1)
                plt.imsave(os.path.join(result_dir_test, 'png', '%04d_output.png' % id), output_, cmap=cmap)