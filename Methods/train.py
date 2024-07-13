import os.path
import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from model import DenseEncoder_Decoder_g2, Decoder_g1, Decoder_g3
from DataLoader.Data_Lsystem import get_dataLsystem_loader
from utils import *
from nltk.translate.bleu_score import corpus_bleu
from DataLoader.Data_Julia import get_data_Julia_loader
from DataLoader.Data_fractal_DB import get_dataDB_loader
import math
import argparse
import json

def main():
    """
    Training and validation.
    """

    global best_bleu4, epochs_since_improvement, data_name, word_map, save_dir, min_loss

    # Read word map
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)


    encoder_decoder_g2 = DenseEncoder_Decoder_g2()
    encoder_decoder_g2_optimizer = torch.optim.Adam(params=encoder_decoder_g2.parameters(), lr=encoder_lr)

    decoder_g1 = Decoder_g1(attention_dim=attention_dim,
                            embed_dim=emb_dim,
                            decoder_dim=decoder_dim,
                            vocab_size=len(word_map),
                            encoder_dim=encoder_dim,
                            dropout=dropout)
    decoder_g1_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder_g1.parameters()),
                                            lr=decoder_lr)

    decoder_g3 = Decoder_g3()

    decoder_g3_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder_g3.parameters()),
                                            lr=encoder_lr)



    # Move to GPU, if available
    encoder_decoder_g2 = encoder_decoder_g2.to(device)
    decoder_g1 = decoder_g1.to(device)
    decoder_g3 = decoder_g3.to(device)
    # Loss function
    criterion_CE = nn.CrossEntropyLoss().to(device)
    criterion_MSE = nn.MSELoss().to(device)
    # Load Data
    train_loader_Lsystem, val_loader_Lsystem,_ = get_dataLsystem_loader(data_folder, data_name, batch_size, workers)
    train_loader_Julia, val_loader_Julia = get_data_Julia_loader(Julia_data_folder, batch_size)
    train_loader_fractal = get_dataDB_loader(fractalDB_data_folder, batch_size)
    # Epochs
    for epoch in range(0, epochs):
        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_g1_optimizer, 0.8)
        # One epoch's training
        train(train_loader_Lsystem=train_loader_Lsystem,
              train_Julia_loader=train_loader_Julia,
              train_fractal60_loader=train_loader_fractal,
              encoder_decoder_g2=encoder_decoder_g2,
              decoder_g1=decoder_g1,
              decoder_g3=decoder_g3,
              encoder_decoder_g2_optimizer=encoder_decoder_g2_optimizer,
              decoder_g1_optimizer=decoder_g1_optimizer,
              decoder_g3_optimizer=decoder_g3_optimizer,
              criterion_CE=criterion_CE,
              criterion_MSE=criterion_MSE,
              epoch=epoch)

        # One epoch's validation
        recent_bleu4, recent_loss = validate(val_loader_Lsystem=val_loader_Lsystem,
                                             val_loader_Julia=val_loader_Julia,
                                             encoder_decoder_g2=encoder_decoder_g2,
                                             decoder_g1=decoder_g1,
                                             criterion_CE=criterion_CE,
                                             criterion_MSE=criterion_MSE,
                                             )

        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(save_dir_seq, data_name, encoder_decoder_g2, decoder_g1,
                        decoder_g3, encoder_decoder_g2_optimizer, decoder_g1_optimizer, decoder_g3_optimizer,
                        is_best, epoch, epochs_since_improvement, recent_bleu4, recent_loss)


    # Next we will only update the decoder g2
    min_loss = 1e5
    checkpoint = torch.load(os.path.join(save_dir_seq, "BEST_checkpoint_fractal_random_1_cap_per_img_1_min_word_freq.pth.tar"))
    encoder_decoder_g2_best = checkpoint['encoder_decoder_g2']
    for p in encoder_decoder_g2_best.adaptive_pool_for_decoder_g2_g3.parameters():
        p.requires_grad = True
    for p in encoder_decoder_g2_best.flatten_for_decoder_g2_g3.parameters():
        p.requires_grad = True
    for p in encoder_decoder_g2_best.Decoder_g2.parameters():
        p.requires_grad = True
    for param in encoder_decoder_g2_best.linear_for_decoder_g3.parameters():
        param.requires_grad = False
    for param in encoder_decoder_g2_best.Encoder.parameters():
        param.requires_grad = False
    for param in encoder_decoder_g2_best.adaptive_pool_for_decoder_g1.parameters():
        param.requires_grad = False
    encoder_decoder_g2_best_optimizer = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, encoder_decoder_g2_best.parameters()),
        lr=encoder_lr)

    for epoch in range(0, 30):
        train_only_decoder_g2(train_Julia_loader=train_loader_Julia,
                              encoder_decoder_g2=encoder_decoder_g2_best,
                              criterion_MSE=criterion_MSE,
                              encoder_decoder_g2_optimizer=encoder_decoder_g2_best_optimizer,
                              epoch=epoch)

        recent_loss = validate_only_decoder_g2(val_loader_Julia=val_loader_Julia,
                                               encoder_decoder_g2=encoder_decoder_g2_best, criterion_MSE=criterion_MSE)
        is_best = recent_loss < min_loss
        min_loss = min(recent_loss, min_loss)
        save_checkpoint(save_dir, data_name, encoder_decoder_g2_best, checkpoint['decoder_g1'],
                        checkpoint['decoder_g3'], encoder_decoder_g2_best_optimizer,
                        checkpoint['decoder_g1_optimizer'], checkpoint['decoder_g3_optimizer'], is_best)


def train(train_loader_Lsystem, train_Julia_loader, train_fractal60_loader, encoder_decoder_g2, decoder_g1, decoder_g3,
          encoder_decoder_g2_optimizer, decoder_g1_optimizer, decoder_g3_optimizer, criterion_CE, criterion_MSE, epoch):
    encoder_decoder_g2.train()
    decoder_g1.train()
    decoder_g3.train()
    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    for batch, (fractal_DB, Julia, L_system) in enumerate(
            zip(train_fractal60_loader, train_Julia_loader, train_loader_Lsystem)):
        # DB的结果
        image_DB = fractal_DB['image'].to(device)
        _, _, latent_DB = encoder_decoder_g2(image_DB)
        pre_DB = decoder_g3(latent_DB)
        loss_DB = criterion_MSE(image_DB, pre_DB)
        # Julia的结果
        image_Julia = Julia['image'].to(device)
        label_Julia = Julia['label'].to(device)
        _, pre, latent_Julia = encoder_decoder_g2(image_Julia)
        pre_Julia = decoder_g3(latent_Julia)
        loss_Julia = criterion_MSE(image_Julia, pre_Julia)
        loss_pre_Julia = criterion_MSE(pre, label_Julia)
        # L_system的结果
        image_L = L_system[0].to(device)
        caps = L_system[1].to(device)
        caplens = L_system[2].to(device)
        imgs, _, latent_L = encoder_decoder_g2(image_L)
        pre_L = decoder_g3(latent_L)
        loss_L = criterion_MSE(image_L, pre_L)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder_g1(imgs, caps, caplens)
        targets = caps_sorted[:, 1:]
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)
        loss = criterion_CE(scores.data, targets.data)
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
        # Compute total loss and do backward
        total_loss_image_decoder = (loss_L + loss_Julia + loss_DB) / 3
        final_loss = 0.05 * loss_pre_Julia + 0.95 * loss + 0.1 * total_loss_image_decoder
        encoder_decoder_g2_optimizer.zero_grad()
        decoder_g1_optimizer.zero_grad()
        decoder_g3_optimizer.zero_grad()
        final_loss.backward()
        if grad_clip is not None:
            clip_gradient(decoder_g1_optimizer, grad_clip)
            clip_gradient(encoder_decoder_g2_optimizer, grad_clip)
        # Update weights
        encoder_decoder_g2_optimizer.step()
        decoder_g1_optimizer.step()
        decoder_g3_optimizer.step()

        # Keep track of metrics
        top5 = accuracy(scores.data, targets.data, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()
        # break
        # Print status
        if batch % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, batch, len(train_loader_Lsystem),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))

            print("epoch:{}, batch:{}, loss_DB:{}, loss_Julia:{}, loss_L:{}, loss_pre_Julia".format(epoch, batch,
                                                                                                    loss_DB.item(),
                                                                                                    loss_Julia.item(),
                                                                                                    loss_L.item()), loss_pre_Julia.item())


def train_only_decoder_g2(train_Julia_loader, encoder_decoder_g2, criterion_MSE, encoder_decoder_g2_optimizer, epoch):
    encoder_decoder_g2.adaptive_pool_for_decoder_g2_g3.train()
    encoder_decoder_g2.Decoder_g2.train()
    encoder_decoder_g2.flatten_for_decoder_g2_g3.train()
    encoder_decoder_g2.Encoder.eval()
    encoder_decoder_g2.adaptive_pool_for_decoder_g1.eval()

    for batch, Julia in enumerate(train_Julia_loader):
        image_Julia = Julia['image'].to(device)
        label_Julia = Julia['label'].to(device)
        _, pre, latent_Julia = encoder_decoder_g2(image_Julia)
        encoder_decoder_g2_optimizer.zero_grad()
        loss_pre_Julia = criterion_MSE(pre, label_Julia)
        loss_pre_Julia.backward()
        encoder_decoder_g2_optimizer.step()
        if batch % print_freq == 0:
            print("epoch:{}, batch:{}, loss_pre_Julia:{}".format(epoch, batch, loss_pre_Julia.item()))


def validate_only_decoder_g2(val_loader_Julia, encoder_decoder_g2, criterion_MSE):
    encoder_decoder_g2.eval()
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for (batch, meta) in enumerate(val_loader_Julia):
            image = meta['image']
            label = meta['label']
            image = image.to(device)
            label = label.to(device)
            _, pre, _ = encoder_decoder_g2(image)
            test_loss = criterion_MSE(pre, label)
            total_loss += test_loss
            total_samples += pre.size(0)
    loss_julia = total_loss / total_samples
    loss_julia = math.sqrt(loss_julia)
    print("Julia test loss: %.8f" % (loss_julia))

    return loss_julia


def validate(val_loader_Lsystem, val_loader_Julia, encoder_decoder_g2, decoder_g1, criterion_CE, criterion_MSE):
    encoder_decoder_g2.eval()  # eval mode
    decoder_g1.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for (batch, meta) in enumerate(val_loader_Julia):
            image = meta['image']
            label = meta['label']
            image = image.to(device)
            label = label.to(device)
            _, pre, _ = encoder_decoder_g2(image)
            test_loss = criterion_MSE(pre, label)
            total_loss += test_loss
            total_samples += pre.size(0)
    loss_julia = total_loss / total_samples
    loss_julia = math.sqrt(loss_julia)
    print("Julia test loss: %.8f" % (loss_julia))

    with torch.no_grad():
        # Batches
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader_Lsystem):

            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)
            allcaps = allcaps.to(device)
            # Forward prop.
            imgs, _, _ = encoder_decoder_g2(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder_g1(imgs, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            # Calculate loss
            loss = criterion_CE(scores.data, targets.data)

            # Add doubly stochastic attention regularization
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores.data, targets.data, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader_Lsystem),
                                                                                batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            # References
            allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)
            assert len(references) == len(hypotheses)
        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)

        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu=bleu4))

    return bleu4, loss_julia


if __name__ == '__main__':



    # 创建解析对象
    parser = argparse.ArgumentParser(description='Example training script')

    # Data parameters
    parser.add_argument('--data_folder', type=str,
                        default='Data_Generation/L_system/100_padding/',
                        help='folder with data files of L-system images')
    parser.add_argument('--data_name', type=str, default='fractal_random_1_cap_per_img_1_min_word_freq',
                        help='base name shared by data files')
    parser.add_argument('--Julia_data_folder', type=str,
                        default='Data_Generation/Julia_Set/Data/',
                        help='folder with data files of Julia images')
    parser.add_argument('--fractalDB_data_folder', type=str,
                        default='Data_Generation/FractalDB/',
                        help='folder with data files of fractalBD images')

    # Model parameters
    parser.add_argument('--emb_dim', type=int, default=512, help='dimension of word embeddings')
    parser.add_argument('--attention_dim', type=int, default=512, help='dimension of attention linear layers')
    parser.add_argument('--decoder_dim', type=int, default=512, help='dimension of decoder RNN')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        help='sets device for model and PyTorch tensors')
    parser.add_argument('--encoder_dim', type=int, default=2208, help='dimension of encoder')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=120,
                        help='number of epochs to train for (if early stopping is not triggered)')
    parser.add_argument('--epochs_since_improvement', type=int, default=0,
                        help='keeps track of number of epochs since there\'s been an improvement in validation BLEU')
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--workers', type=int, default=1, help='for data-loading; right now, only 1 works with h5py')
    parser.add_argument('--encoder_lr', type=float, default=1e-4, help='learning rate for encoder if fine-tuning')
    parser.add_argument('--decoder_lr', type=float, default=4e-4, help='learning rate for decoder')
    parser.add_argument('--grad_clip', type=float, default=5., help='clip gradients at an absolute value of')
    parser.add_argument('--alpha_c', type=float, default=1.,
                        help='regularization parameter for doubly stochastic attention')
    parser.add_argument('--best_bleu4', type=float, default=0., help='BLEU-4 score right now')
    parser.add_argument('--min_loss', type=float, default=1e5, help='best accuracy so far')
    parser.add_argument('--print_freq', type=int, default=100, help='print training/validation stats every __ batches')
    parser.add_argument('--save_dir_seq', type=str, default='checkpoint/Dense_seq',
                        help='directory to save sequence model checkpoints')
    parser.add_argument('--save_dir', type=str, default='checkpoint/Dense_final', help='overall save directory')

    # 解析命令行参数
    args = parser.parse_args()

    data_folder = args.data_folder
    data_name = args.data_name
    Julia_data_folder = args.Julia_data_folder
    fractalDB_data_folder = args.fractalDB_data_folder

    emb_dim = args.emb_dim
    attention_dim = args.attention_dim
    decoder_dim = args.decoder_dim

    encoder_dim = args.encoder_dim
    dropout = args.dropout
    device = args.device
    epochs = args.epochs
    epochs_since_improvement = args.epochs_since_improvement
    batch_size = args.batch_size
    workers = args.workers
    encoder_lr = args.encoder_lr
    decoder_lr = args.decoder_lr
    grad_clip = args.grad_clip
    alpha_c = args.alpha_c
    best_bleu4 = args.best_bleu4
    min_loss = args.min_loss


    print_freq = args.print_freq
    save_dir_seq = args.save_dir_seq
    save_dir = args.save_dir
    cudnn.benchmark = True

    main()
