# 这里面需要有一个Image Extractor，还需要有一个Image Decoder，还需要有一个caption的deoceder
import torch
from torch import nn
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VITEncoder(nn.Module):
    """
        Encoder.
        """

    def __init__(self, encoded_image_size=14):
        super(VITEncoder, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = torchvision.models.vit_b_16(pretrained=True)  # pretrained ImageNet ResNet-101

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.adaptive_pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.pre_linear = nn.Linear(768, 4)
        self.image_linear = nn.Linear(768, 100)

        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        # print(self.resnet)
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out2 = self.adaptive_pool2(out)
        out2 = self.flatten(out2)
        pre_non_seque = self.pre_linear(out2)
        latent = self.image_linear(out2)
        # 这里的图片需要进一步分改一下
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out, pre_non_seque, latent

    def fine_tune(self):

        for name, param in self.resnet.named_parameters():
            param.requires_grad = True
            # print(name)
            # if "denselayer24" in name or "denselayer23" in name or "11.weight" == name or "11.bias" == name :
            #     param.requires_grad = True
            #     print(name)
        for p in self.adaptive_pool.parameters():
            p.requires_grad = True
        for p in self.adaptive_pool2.parameters():
            p.requires_grad = True
        for p in self.image_linear.parameters():
            p.requires_grad = True
        for p in self.pre_linear.parameters():
            p.requires_grad = True
        for p in self.flatten.parameters():
            p.requires_grad = True

class MobileCombineEncoder(nn.Module):
    """
        Encoder.
        """

    def __init__(self, encoded_image_size=14, out_feature= 4):
        super(MobileCombineEncoder, self).__init__()
        self.enc_image_size = encoded_image_size

        vgg = torchvision.models.mobilenet_v3_small(pretrained=True)  # pretrained ImageNet ResNet-101

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(vgg.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        # Assuming self.linear is initially set as follows
        self.linear = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size = (1,1)),
            nn.Flatten(),
            nn.Linear(576, 1024),
            nn.Hardswish(),
            nn.Dropout(0.2),

        )
        self.pre_linear = nn.Linear(1024, 4)
        self.image_linear = nn.Linear(1024, 100)

        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        # print(self.A(out).shape)
        out2 = self.linear(out)
        pre_non_seque = self.pre_linear(out2)
        image_decode = self.image_linear(out2)
        # 这里的图片需要进一步分改一下
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out, pre_non_seque, image_decode

    def fine_tune(self, fine_tune=True):
        #
        # for name, param in self.resnet.named_parameters():
        #     param.requires_grad = True
        #     # print(name)
        #     if "0.11" in name or "0.12" in name:
        #         param.requires_grad = True
        #         print(name)
        # for p in self.linear.parameters():
        #     p.requires_grad = True

        for p in self.resnet.parameters():
            p.requires_grad = True
        for p in self.linear.parameters():
            p.requires_grad = True
        for p in self.adaptive_pool.parameters():
            p.requires_grad = True
        for p in self.image_linear.parameters():
            p.requires_grad = True
        for p in self.pre_linear.parameters():
            p.requires_grad = True

class VGGCombineEncoder(nn.Module):
    """
        Encoder.
        """

    def __init__(self, encoded_image_size=14, out_feature= 4):
        super(VGGCombineEncoder, self).__init__()
        self.enc_image_size = encoded_image_size

        vgg = torchvision.models.vgg16(pretrained=True)  # pretrained ImageNet ResNet-101

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(vgg.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        # Assuming self.linear is initially set as follows
        self.linear = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size = (7,7)),
            nn.Flatten(),
            nn.Linear(25088, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),

        )
        self.pre_linear = nn.Linear(4096,4)
        self.image_linear = nn.Linear(4096,100)
        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out2 = self.linear(out)
        pre_non_seque = self.pre_linear(out2)
        image_decode = self.image_linear(out2)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out, pre_non_seque, image_decode

    def fine_tune(self, fine_tune=True):
        # name_ls = ['0.26.weight', '0.26.bias', '0.28.weight', '0.28.bias']
        # for name, param in self.resnet.named_parameters():
        #     param.requires_grad = True
        #     if name in name_ls:
        #         param.requires_grad = True
        for p in self.resnet.parameters():
            p.requires_grad = True
        for p in self.linear.parameters():
            p.requires_grad = True
        for p in self.adaptive_pool.parameters():
            p.requires_grad = True
        for p in self.image_linear.parameters():
            p.requires_grad = True
        for p in self.pre_linear.parameters():
            p.requires_grad = True

class ResnetCombineEncoder(nn.Module):
    """
        Encoder.
        """

    def __init__(self, encoded_image_size=14, out_feature= 4):
        super(ResnetCombineEncoder, self).__init__()
        self.enc_image_size = encoded_image_size
        resnet = torchvision.models.resnet50(pretrained=True)  # pretrained ImageNet ResNet-101
        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        # Assuming self.linear is initially set as follows
        infeatures = resnet.fc.in_features
        self.linear = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size = (1,1)),
            nn.Flatten(),
            nn.Linear(infeatures, 256),
            nn.ReLU(),

        )
        # self.adaptive_pool2 = nn.AdaptiveAvgPool2d((1,1))
        # self.flatten = nn.Flatten(),
        # self.linear_common = nn.Linear(infeatures, 256)
        # self.Relu = nn.ReLU()

        self.pre_linear = nn.Linear(256, 4)
        self.image_linear = nn.Linear(256, 100)

        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out2 = self.linear(out)
        pre_non_seque = self.pre_linear(out2)
        latent = self.image_linear(out2)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        # 如果是non-sequence task的话就是输出对应值就好了

        return out, pre_non_seque, latent

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = True
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4

        for p in self.linear.parameters():
            p.requires_grad = True
        
        for p in self.adaptive_pool.parameters():
            p.requires_grad = True
        
        for p in self.image_linear.parameters():
            p.requires_grad = True
            
        for p in self.pre_linear.parameters():
            p.requires_grad = True

class DenseCombineEncoder(nn.Module):
    """
        Encoder.
        """

    def __init__(self, encoded_image_size=14, out_feature= 4):
        super(DenseCombineEncoder, self).__init__()
        self.enc_image_size = encoded_image_size

        vgg = torchvision.models.densenet161(pretrained=True)  # pretrained ImageNet ResNet-101
        # Remove linear and pool layers (since we're not doing classification)
        modules = vgg.features[:12]
        self.resnet = nn.Sequential(*modules)
        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        # Assuming self.linear is initially set as follows
        self.adaptive_pool2 = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.pre_linear = nn.Linear(2208, 4)
        self.image_linear = nn.Linear(2208, 100)
        # self.linear = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Flatten(),
        #     nn.Linear(2208,4)
        # )
        # self.image_encoder = nn.Sequential
        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        # print(self.resnet)
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out2 = self.adaptive_pool2(out)
        out2 = self.flatten(out2)
        pre_non_seque = self.pre_linear(out2)
        latent = self.image_linear(out2)
        # 这里的图片需要进一步分改一下
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out, pre_non_seque, latent

    def fine_tune(self):

        for name, param in self.resnet.named_parameters():
            param.requires_grad = True
            # print(name)
            # if "denselayer24" in name or "denselayer23" in name or "11.weight" == name or "11.bias" == name :
            #     param.requires_grad = True
            #     print(name)
        for p in self.adaptive_pool.parameters():
            p.requires_grad = True
        for p in self.adaptive_pool2.parameters():
            p.requires_grad = True
        for p in self.image_linear.parameters():
            p.requires_grad = True
        for p in self.pre_linear.parameters():
            p.requires_grad = True
        for p in self.flatten.parameters():
            p.requires_grad = True

# encoder = VITCombineEncoder(encoded_image_size=14, out_feature=4)
#
# # 生成一个随机的图像作为测试输入
# dummy_input = torch.randn(4, 3, 224, 224)  # 1张3通道的256x256大小的图像
#
# # 运行模型
# output, pre_non_seque = encoder(dummy_input)
# encoder.forward(dummy_input)
# # 打印输出的维度
# print("Encoded Image Dimensions:", output.shape)
# print("Pre Non-sequence Dimensions:", pre_non_seque.shape)
# 首先这里的encoder部分可以自行更换（包括resnet101等等）
class ImageDecoder(nn.Module):
    def __init__(self):
        super(ImageDecoder, self).__init__()

        ## 模型中间块儿
        def block(in_feat, out_feat, normalize=True):  ## block(in， out )
            layers = [nn.Linear(in_feat, out_feat)]  ## 线性变换将输入映射到out维
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))  ## 正则化
            layers.append(nn.LeakyReLU(0.2, inplace=True))  ## 非线性激活函数
            return layers

        ## prod():返回给定轴上的数组元素的乘积:1*28*28=784
        self.model = nn.Sequential(
            *block(100, 128, normalize=False),  ## 线性变化将输入映射 100 to 128, 正则化, LeakyReLU
            *block(128, 256),  ## 线性变化将输入映射 128 to 256, 正则化, LeakyReLU
            *block(256, 512),  ## 线性变化将输入映射 256 to 512, 正则化, LeakyReLU
            *block(512, 1024),  ## 线性变化将输入映射 512 to 1024, 正则化, LeakyReLU
            nn.Linear(1024, 256*256*3),  ## 线性变化将输入映射 1024 to 784
            nn.Tanh()  ## 将(784)的数据每一个都映射到[-1, 1]之间
        )

    ## view():相当于numpy中的reshape，重新定义矩阵的形状:这里是reshape(64, 1, 28, 28)
    def forward(self, z):  ## 输入的是(64， 100)的噪声数据
        imgs = self.model(z)  ## 噪声数据通过生成器模型
        img_shape=(3, 256, 256)
        imgs = imgs.view(imgs.size(0), *img_shape)  ## reshape成(64, 1, 28, 28)
        return imgs  ## 输出为64张大小为(1, 28, 28)的图像



class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image，对图片进行展开
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding，对这里的caption进行embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, enc0oder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
