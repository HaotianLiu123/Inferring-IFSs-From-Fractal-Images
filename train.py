import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from Model import DenseCombineEncoder, DecoderWithAttention, ImageDecoder
from DataLoader.datasets import *
from DataLoader.utils import *
from nltk.translate.bleu_score import corpus_bleu
from DataLoader.Data_Julia import get_data_Julia_loader
from DataLoader.Data_fractal_DB import get_dataDB_loader
import math

# Data parameters
# data_folder = '/home/haotian/Fractal/Data/Fractal_fixed_category/100padding_modify/'  # folder with data files saved by create_input_files.py
#data_folder = '/home/haotian/Work/Data/Fixed_Category/100padding_modify/'
data_folder = "Data Generation/L-System/100_padding"
data_name = 'fractal_random_1_cap_per_img_1_min_word_freq'  # base name shared by data files

# Model parameters
emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
start_epoch = 0
epochs = 120  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 16
workers = 1  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu4 = 0.  # BLEU-4 score right now
best_acc = 1e5
print_freq = 100  # print training/validation stats every __ batches
fine_tune_encoder = False  # fine-tune encoder?
checkpoint = None  # path to checkpoint, None if none
# save_dir_seq = 'fixed_category_combine/Dense_combine_seq2'
# save_dir_non = 'fixed_category_combine/Dense_combine_non2'
# save_dir = 'fixed_category_combine/Dense_combine2'
encoder_dim = 2208
save_dir_seq = 'checkpoints/Dense/'
save_dir_non = 'checkpoints/Dense/'
save_dir = 'checkpoints/Dense/'
alpha = 0.05
belta = 0.1


def main():
    """
    Training and validation.
    """

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map, save_dir, best_acc, encoder_dim, alpha

    # Read word map
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # Initialize / load checkpoint
    if checkpoint is None:
        decoder = DecoderWithAttention(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(word_map),
                                       encoder_dim=encoder_dim,
                                       dropout=dropout)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)

        encoder = DenseCombineEncoder()
        encoder_optimizer = torch.optim.Adam(params=encoder.parameters(), lr=encoder_lr)
        # encoder.fine_tune()
        # encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
        #                                      lr=encoder_lr)
        Image_decoder = ImageDecoder()

        Image_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, Image_decoder.parameters()),
                                           lr=encoder_lr)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)

    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)
    Image_decoder = Image_decoder.to(device)
    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    train_loader_Julia, val_loader_Julia = get_data_Julia_loader()
    train_loader_fractal = get_dataDB_loader()
    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)
        if epoch >= 30:
            alpha = 0.3
        # One epoch's training
        train(train_loader=train_loader,
              train_Julia_loader=train_loader_Julia,
              train_fractal60_loader=train_loader_fractal,
              encoder=encoder,
              decoder=decoder,
              image_decoder=Image_decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              imagedecoder_optimizer=Image_optimizer,
              epoch=epoch)

        # One epoch's validation
        recent_bleu4, recent_accuracy = validate(val_loader=val_loader,
                                                 val_loader_Julia=val_loader_Julia,
                                                 encoder=encoder,
                                                 decoder=decoder,
                                                 criterion=criterion,
                                                 epoch=epoch
                                                 )

        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        # best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            # epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(save_dir_seq, data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_bleu4, is_best)

        is_best = recent_accuracy < best_acc
        # best_acc = min(recent_accuracy, best_acc)
        if not is_best:
            # epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(save_dir_non, data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_accuracy, is_best)

        is_best = recent_accuracy < best_acc and recent_bleu4 > best_bleu4
        best_acc = min(recent_accuracy, best_acc)
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(save_dir, data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_bleu4, is_best)


def train(train_loader, train_Julia_loader, train_fractal60_loader, encoder, image_decoder, decoder, criterion,
          encoder_optimizer,
          decoder_optimizer, imagedecoder_optimizer, epoch):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()
    image_decoder.train()
    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()
    criterion_Julia = nn.MSELoss().to(device)
    if epoch % 1 == 0:
        encoder.train()
        image_decoder.train()
        decoder.train()
        for batch, (fractal_DB, Julia, L_system) in enumerate(
                zip(train_fractal60_loader, train_Julia_loader, train_loader)):
            # DB的结果
            image_DB = fractal_DB['image'].to(device)
            _, _, latent_DB = encoder(image_DB)
            pre_DB = image_decoder(latent_DB)
            loss_DB = criterion_Julia(image_DB, pre_DB)
            # Julia的结果
            image_Julia = Julia['image'].to(device)
            label_Julia = Julia['label'].to(device)
            _, pre, latent_Julia = encoder(image_Julia)
            pre_Julia = image_decoder(latent_Julia)
            loss_Julia = criterion_Julia(image_Julia, pre_Julia)
            loss_pre_Julia =criterion_Julia(pre, label_Julia)
            # L_system的结果
            image_L = L_system[0].to(device)
            caps =  L_system[1].to(device)
            caplens = L_system[2].to(device)
            imgs, _, latent_L = encoder(image_L)
            pre_L = image_decoder(latent_L)
            loss_L = criterion_Julia(image_L, pre_L)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)
            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]
            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)
            # 使用 pad_packed_sequence 还原为原始张量
            # scores, _ = pad_packed_sequence(scores, batch_first=True)
            # targets, _ = pad_packed_sequence(targets, batch_first=True)

            # Calculate loss
            loss = criterion(scores.data, targets.data)
            # Add doubly stochastic attention regularization
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
            total_loss_image_decoder = (loss_L + loss_Julia + loss_DB) / 3
            final_loss = alpha * loss_pre_Julia + (1-alpha) * loss + belta*total_loss_image_decoder
            encoder_optimizer.zero_grad()
            imagedecoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            final_loss.backward()

            # Clip gradients
            if grad_clip is not None:
                clip_gradient(decoder_optimizer, grad_clip)
                if encoder_optimizer is not None:
                    clip_gradient(encoder_optimizer, grad_clip)
            # Update weights
            decoder_optimizer.step()
            # if encoder_optimizer is not None:
            encoder_optimizer.step()
            imagedecoder_optimizer.step()

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
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, batch, len(train_loader),
                                                                              batch_time=batch_time,
                                                                              data_time=data_time, loss=losses,
                                                                              top5=top5accs))


                print("epoch:{}, batch:{}, loss_DB:{}, loss_Julia:{}, loss_L:{}, loss_pre_Julia".format(epoch, batch, loss_DB.item(),
                                                                                    loss_Julia.item(), loss_L.item()), loss_pre_Julia.item())


def validate(val_loader, val_loader_Julia, encoder, decoder, criterion, epoch):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    correct_count = 0
    total_samples = 0
    MSE_loss = nn.MSELoss()

    if epoch % 1 == 0:
        with torch.no_grad():
            for (batch, meta) in enumerate(val_loader_Julia):
                image = meta['image']
                label = meta['label']
                image = image.to(device)
                label = label.to(device)
                _, pre, _ = encoder(image)
                test_loss = MSE_loss(pre, label)
                correct_count += test_loss
                total_samples += pre.size(0)
                # break
        accuracy_julia = correct_count / total_samples
        accuracy_julia = math.sqrt(accuracy_julia)
        print("test accuracy: %.8f" % (accuracy_julia))
    else:
        accuracy_julia = 1e5
    if epoch % 1 == 0:
        # explicitly disable gradient calculation to avoid CUDA memory error
        # solves the issue #57
        with torch.no_grad():
            # Batches
            for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

                # Move to device, if available
                imgs = imgs.to(device)
                caps = caps.to(device)
                caplens = caplens.to(device)
                allcaps = allcaps.to(device)
                # Forward prop.
                if encoder is not None:
                    imgs, _, _ = encoder(imgs)
                scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

                # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
                targets = caps_sorted[:, 1:]

                # Remove timesteps that we didn't decode at, or are pads
                # pack_padded_sequence is an easy trick to do this
                scores_copy = scores.clone()
                scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
                targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)

                # Calculate loss
                loss = criterion(scores.data, targets.data)

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
                          'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader),
                                                                                    batch_time=batch_time,
                                                                                    loss=losses, top5=top5accs))

                # Store references (true captions), and hypothesis (prediction) for each image
                # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
                # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

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
                # break
            # Calculate BLEU-4 scores
            bleu4 = corpus_bleu(references, hypotheses)

            print(
                '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                    loss=losses,
                    top5=top5accs,
                    bleu=bleu4))
    else:
        bleu4 = 0.0
    return bleu4, accuracy_julia


if __name__ == '__main__':
    main()
