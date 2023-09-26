from transformers import AutoModel, AutoTokenizer, AutoConfig
from torch import nn
import torch
import math
from torch.nn.init import xavier_uniform_

from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm

from copy import deepcopy as cp

from load_bert import load_bert_model

class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        #print (x.shape, self.pe[:x.size(0), :].shape)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class SequenceClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout_p=0.1):
        super().__init__()
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(hidden_size, num_labels)

        self._init_weights()

    def forward(self, pooled_output):
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits

    def _init_weights(self):
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()

class ActionPredictor(torch.nn.Module):
    def __init__(self, d_model=768, num_actions=8):
        super(ActionPredictor, self).__init__()
        self.action_predictor = nn.Linear(d_model, num_actions)
        #self.init_weights()

    def init_weights(self):
        initrange = 0.1

        self.action_predictor.bias.data.zero_()
        self.action_predictor.weight.data.uniform_(-initrange, initrange)

    def forward(self, state_tensor):
        actions = torch.nn.Softmax(-1)(self.action_predictor(state_tensor))
        return actions
        
class FineTunedModel(torch.nn.Module):
    def __init__(self, tasks, label_nums, config, pretrained_model_name='bert-base-uncased', tf_checkpoint=None, \
                 dropout=0.1):
        super(FineTunedModel, self).__init__()

        self.config = config

        if tf_checkpoint is None:
            self.encoder = AutoModel.from_pretrained(pretrained_model_name, config=config)
        else:
            self.encoder = load_bert_model(tf_checkpoint)

        self.drop = nn.Dropout(dropout)
        
        self.output_heads = nn.ModuleDict()
        for task in tasks:
            decoder = SequenceClassificationHead(self.encoder.config.hidden_size, label_nums[task.lower()])
            # ModuleDict requires keys to be strings
            self.output_heads[task.lower()] = decoder
        
        self.state_predictor = nn.Linear(config.hidden_size, 1)
        
        #self.init_weights()

    def forward(self, task_name, src=None, mask=None, token_type_ids=None, pooled_output=None, discriminator=False, output_hidden_states=True):
        
        if discriminator == False:
            outputs = self.encoder(
                src,
                attention_mask=mask,
                token_type_ids=token_type_ids,output_hidden_states=output_hidden_states)
            
            encoder_output = outputs[0]
            pooled_output = outputs[1]
        
        #encoder_output = self.encoder(src)
        out = self.output_heads[task_name.lower()](pooled_output)
        if task_name == 'sts-b':
            out = nn.ReLU()(out)
        
        model_state = self.state_predictor(self.encoder.pooler.dense.weight).reshape(1,-1)
        
        features = None

        if output_hidden_states == True and discriminator == False:
            features = torch.cat(outputs[-1][1:-1], dim=0).view(self.config.num_hidden_layers - 1,
                                                                 -1,
                                                                 src.size()[1],
                                                                 self.config.hidden_size)[:, :, 0]
        
        if discriminator == False:
            return (out, features, model_state, pooled_output) #encoder_output
        else:
            return (out, pooled_output, model_state, pooled_output)
        
class ReZeroEncoderLayer(Module):
    r"""ReZero-TransformerEncoderLayer is made up of self-attn and feedforward network.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        use_LayerNorm: using either no LayerNorm (dafault=False), or use LayerNorm "pre", or "post"

    Examples::
        >>> encoder_layer = ReZeroEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation = "relu", 
                 use_LayerNorm = False, init_resweight = 0, resweight_trainable = True):
        super(ReZeroEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Define the Resisdual Weight for ReZero
        self.resweight = torch.nn.Parameter(torch.Tensor([init_resweight]), requires_grad = resweight_trainable)

        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.use_LayerNorm = use_LayerNorm
        if self.use_LayerNorm != False:
            self.norm1 = LayerNorm(d_model)
            self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == "tanh":
            self.activation = torch.tanh

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(ReZeroEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        src2 = src
        if self.use_LayerNorm == "pre":
            src2 = self.norm1(src2)
        src2 = self.self_attn(src2, src2, src2, attn_mask=src_mask,key_padding_mask=src_key_padding_mask)[0]
        # Apply the residual weight to the residual connection. This enables ReZero.
        src2 = self.resweight * src2
        src2 = self.dropout1(src2)
        if self.use_LayerNorm == False:
            src = src + src2
        elif self.use_LayerNorm == "pre":
            src = src + src2
        elif self.use_LayerNorm == "post":
            src = self.norm1(src + src2)
        src2 = src
        if self.use_LayerNorm == "pre":
            src2 = self.norm1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src2 = self.resweight * src2
        src2 = self.dropout2(src2)
        if self.use_LayerNorm == False:
            src = src + src2
        elif self.use_LayerNorm == "pre":
            src = src + src2
        elif self.use_LayerNorm == "post":
            src = self.norm1(src + src2)
        return src
    
class TransformerModel(torch.nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.1, 
                 encoder_version = 'ReZero', pooling_method='cls'):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pooling_method = pooling_method
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        if encoder_version == 'ReZero':
            encoder_layers = ReZeroEncoderLayer(ninp, nhead, nhid, dropout, 
                activation = "relu", use_LayerNorm = False, init_resweight = 0, 
                resweight_trainable = True)
        elif encoder_version == 'pre':
            encoder_layers = ReZeroEncoderLayer(ninp, nhead, nhid, dropout, 
                activation = "relu", use_LayerNorm = 'pre', init_resweight = 1, 
                resweight_trainable = False)
        elif encoder_version == 'post':
            encoder_layers = ReZeroEncoderLayer(ninp, nhead, nhid, dropout, 
                activation = "relu", use_LayerNorm = 'post', init_resweight = 1, 
                resweight_trainable = False)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = torch.nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.pooler = torch.nn.Linear(nhid, nhid)
        self._reset_parameters()
        self.init_weights()
        
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.pooler.bias.data.zero_()
        self.pooler.weight.data.uniform_(-initrange, initrange)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)

        if self.pooling_method == 'cls':
            encoder_output = output[0]
        elif self.pooling_method == 'mean':
            encoder_output = output.mean(0)
        
        pooler_output = self.pooler(encoder_output)

        return output, pooler_output
    
class FineTunedTransformerModel(torch.nn.Module):
    def __init__(self, ntokens, emsize, nhid, nlayers, tasks, label_nums, pooling_method='cls', \
                 encoder_version='post', nhead=8, dropout=0.1):
        super(FineTunedTransformerModel, self).__init__()

        self.encoder = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, 
                             dropout, encoder_version = encoder_version, pooling_method=pooling_method)

        self.drop = nn.Dropout(dropout)

        self.output_heads = nn.ModuleDict()
        for task in tasks:
            decoder = SequenceClassificationHead(nhid, label_nums[task.lower()])
            # ModuleDict requires keys to be strings
            self.output_heads[task.lower()] = decoder
        
        self.state_predictor = nn.Linear(nhid, 1)
        
        #self.init_weights()

    def forward(self, task_name, src=None, mask=None, token_type_ids=None, pooled_output=None, discriminator=False):
        
        out = self.encoder(src.transpose(0,1))

        encoder_output = out[0]
        pooled_output = out[1]

        #encoder_output = self.encoder(src)
        out = self.output_heads[task_name.lower()](pooled_output)
        if task_name == 'sts-b':
            out = nn.ReLU()(out)
        
        model_state = self.state_predictor(self.encoder.pooler.weight).reshape(1,-1)

        return out, encoder_output.transpose(0,1), model_state, pooled_output
