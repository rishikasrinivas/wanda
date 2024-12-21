import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import numpy as np
import collections
from typing import Union
from transformers import AutoTokenizer, AutoModel

#https://github.com/lecode-official/pytorch-lottery-ticket-hypothesis/blob/main/source/lth/models/__init__.py
class Layer:
    """Represents a single prunable layer in the neural network."""

    def __init__(
            self,
            name: str,
            weights: torch.nn.Parameter,
            initial_weights: torch.Tensor,
            pruning_mask: torch.Tensor) -> None:
        
        """Initializes a new Layer instance.

        Args:
            name (str): The name of the layer.
            kind (LayerKind): The kind of the layer.
            weights (torch.nn.Parameter): The weights of the layer.
            biases (torch.nn.Parameter): The biases of the layer.
            initial_weights (torch.Tensor): A copy of the initial weights of the layer.
            initial_biases (torch.Tensor): A copy of the initial biases of the layer.
            pruning_mask (torch.Tensor): The current pruning mask of the layer.
        """

        self.name = name
        self.weights = weights
        self.initial_weights = initial_weights
        self.pruning_mask = pruning_mask

# referenced from: https://github.com/lecode-official/pytorch-lottery-ticket-hypothesis/blob/main/source/lth/models/__init__.py
class BaseModel(torch.nn.Module):
    """Represents the base class for all models."""

    def __init__(self) -> None:
        """Initializes a new BaseModel instance. Since this is a base class, it should never be called directly."""

        # Invokes the constructor of the base class
        super().__init__()

        # Initializes some class members
        self.layers = None


    def initialize(self) -> None:
        """Initializes the model. It initializes the weights of the model using Xavier Normal (equivalent to Gaussian Glorot used in the original
        Lottery Ticket Hypothesis paper). It also creates an initial pruning mask for the layers of the model. These are initialized with all ones. A
        pruning mask with all ones does nothing. This method must be called by all sub-classes at the end of their constructor.
        """


        # Gets the all the fully-connected and convolutional layers of the model (these are the only ones that are being used right now, if new layer
        # types are introduced, then they have to be added here, but right now all models only consist of these two types)
        self.layers = []
        for parameter_name, parameter in self.named_parameters():
            weights = parameter
            
            
            weights.requires_grad = True
            init_weights=parameter.clone()

            # Initializes the pruning masks of the layer, which are used for pruning as well as freezing the pruned weights during training
            pruning_mask = torch.ones_like(init_weights, dtype=torch.uint8).to('cuda')  # pylint: disable=no-member
            # Adds the layer to the internal list of layers


            self.layers.append(Layer(parameter_name, weights, init_weights, pruning_mask))
        

    def get_layer_names(self):
        """Retrieves the internal names of all the layers of the model.

        Returns:
            list[str]: Returns a list of all the names of the layers of the model.
        """

        layer_names = []
        for layer in self.layers:
            layer_names.append(layer.name)
        return layer_names

    def get_layer(self, layer_name: str) -> Layer:
        """Retrieves the layer of the model with the specified name.

        Args:
            layer_name (str): The name of the layer that is to be retrieved.

        Raises:
            LookupError: If the layer does not exist, an exception is raised.

        Returns:
            Layer: Returns the layer with the specified name.
        """

        for layer in self.layers:
            if layer.name == layer_name:
                return layer
        raise LookupError(f'The specified layer "{layer_name}" does not exist.')

    def update_layer_weights(self, mask, layer_name: str, new_weights: torch.Tensor) -> None:
        """Updates the weights of the specified layer.

        Args:
            layer_name (str): The name of the layer whose weights are to be updated.
            new_weights (torch.Tensor): The new weights of the layer.
        """


        with torch.no_grad():
            # Update the layer weights
            self.state_dict()[layer_name].copy_(new_weights)
            self.get_layer(layer_name).weights.copy_(new_weights)

            self.get_layer(layer_name).pruning_mask.copy_(mask)

    def get_total_num_weights(self):
        terms =0
        for l in self.layers:
            l = self.get_layer(l.name)
            terms += l.weights.flatten().shape[0]
        return terms

    def reset(self) -> None:
        """Resets the model back to its initial initialization."""

        for layer in self.layers:
            self.state_dict()[f'{layer.name}.weight'].copy_(layer.initial_weights)

    def move_to_device(self, device: Union[int, str, torch.device]) -> None:  # pylint: disable=no-member
        """Moves the model to the specified device.

        Args:
            device (Union[int, str, torch.device]): The device that the model is to be moved to.
        """

        # Moves the model itself to the device
        self.to(device)

        # Moves the initial weights, initial biases, and the pruning masks also to the device
        for layer in self.layers:
            layer.initial_weights = layer.initial_weights.to(device)
            layer.pruning_mask = layer.pruning_mask.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass through the neural network. Since this is the base model, the method is not implemented and must be implemented
        in all classes that derive from the base model.

        Args:
            x (torch.Tensor): The input to the neural network.

        Raises:
            NotImplementedError: _description_

        Returns:
            torch.Tensor: Returns the output of the neural network.
        """

        raise NotImplementedError()

class BertEntailmentClassifier(BaseModel):
    def __init__(self, encoder_name="bert-base-uncased", vocab=None, freeze_bert=False):
        super().__init__()
        self.vocab = vocab
        self.encoder_name = encoder_name
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.config = self.encoder.config
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        
        self.seqlen=self.config.max_position_embeddings
#         if freeze_bert:
#             for param in self.encoder.parameters():
#                 param.requires_grad = False

        self.encoder_dim = self.encoder.config.hidden_size
        self.mlp_input_dim = self.encoder_dim * 4
        self.dropout = nn.Dropout(0.1)
        self.bn = nn.BatchNorm1d(self.mlp_input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.mlp_input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 3),
        )
        self.output_dim = 3
        self.initialize()
       

    def forward(self, s1, s1len, s2, s2len):
        device = s1.device

        s1 = s1.transpose(1, 0)
        s2 = s2.transpose(1, 0)

        s1_tokens = self.indices_to_bert_tokens(s1)
        s2_tokens = self.indices_to_bert_tokens(s2)

        s1_tokens = {k: v.to(device) for k, v in s1_tokens.items()}
        s2_tokens = {k: v.to(device) for k, v in s2_tokens.items()}

        s1enc = self.encode_sentence(s1_tokens)
        s2enc = self.encode_sentence(s2_tokens)

        diffs = s1enc - s2enc
        prods = s1enc * s2enc

        mlp_input = torch.cat([s1enc, s2enc, diffs, prods], 1)
        mlp_input = self.bn(mlp_input)
   
        mlp_input = self.dropout(mlp_input)
        preds = self.mlp(mlp_input)

        return preds

    def get_final_reprs(self, s1, s1len, s2, s2len):
        device = s1.device

        s1 = s1.transpose(1, 0)
        s2 = s2.transpose(1, 0)

        s1_tokens = self.indices_to_bert_tokens(s1)
        s2_tokens = self.indices_to_bert_tokens(s2)

        s1_tokens = {k: v.to(device) for k, v in s1_tokens.items()}
        s2_tokens = {k: v.to(device) for k, v in s2_tokens.items()}

        s1enc = self.encode_sentence(s1_tokens)
        s2enc = self.encode_sentence(s2_tokens)

        diffs = s1enc - s2enc
        prods = s1enc * s2enc

        mlp_input = torch.cat([s1enc, s2enc, diffs, prods], 1)
        mlp_input = self.bn(mlp_input)
        mlp_input = self.dropout(mlp_input)
        rep = self.mlp[:-1](mlp_input)

        return rep

    def forward_from_final(self, rep):
        preds = self.mlp[-1:](rep)
        return preds

    def indices_to_bert_tokens(self, indices):
        batch_size, seq_len = indices.shape
        words = []
        for i in range(batch_size):
            sentence = []
            for idx in indices[i]:
                if idx.item() in self.vocab['itos']:
                    word = self.vocab['itos'][idx.item()]
                    if word not in ("[PAD]", "<pad>", "PAD"):
                        sentence.append(word)
                else:
                    break
            words.append(sentence)

        return self.tokenizer(words, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True)

    def encode_sentence(self, tokens):
        outputs = self.encoder(**tokens)
        return outputs.last_hidden_state[:, 0, :]

    def to(self, device):
        self.encoder = self.encoder.to(device)
        return super().to(device)


class BowmanEntailmentClassifier(BaseModel):
    """
    The RNN-based entailment model of Bowman et al 2017
    """

    def __init__(self, encoder):
        super().__init__()

        self.encoder = encoder
        self.encoder_dim = encoder.output_dim
        self.mlp_input_dim = self.encoder_dim * 4
        self.dropout = nn.Dropout(0.1)
        self.bn = nn.BatchNorm1d(self.mlp_input_dim)
        self.prune_mask= torch.ones(1024,self.mlp_input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.mlp_input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),  # Mimic classifier MLP keep rate of 94%
            nn.Linear(1024, 3),
        )
        #self.mlp[:-1][0] = prune.ln_structured(self.mlp[:-1][0], name="weight", amount=0.05, dim=1, n=float('-inf'))
        self.output_dim = 3

        self.initialize()


    def forward(self, s1, s1len, s2, s2len):
        s1enc = self.encoder(s1, s1len)
        s2enc = self.encoder(s2, s2len)


        diffs = s1enc - s2enc
        prods = s1enc * s2enc

        mlp_input = torch.cat([s1enc, s2enc, diffs, prods], 1) #1x2048

        mlp_input = self.bn(mlp_input)
        mlp_input = self.dropout(mlp_input)

        preds = self.mlp(mlp_input)

        return preds

    def check_pruned(self, layer='default'):
        if layer == 'default':
            layer = self.mlp[:-1]
        return prune.is_pruned(layer)

    # from https://github.com/jankrepl/mildlyoverfitted/blob/master/github_adventures/lottery/utils.py
    def copy_weights_linear(linear_unpruned, linear_pruned):
        """Copy weights from an unpruned model to a pruned model.

        Modifies `linear_pruned` in place.

        Parameters
        ----------
        linear_unpruned : nn.Linear
            Linear model with a bias that was not pruned.

        linear_pruned : nn.Linear
            Linear model with a bias that was pruned.
        """
        assert check_pruned_linear(linear_pruned)
        assert not check_pruned_linear(linear_unpruned)

        with torch.no_grad():
            linear_pruned.weight_orig.copy_(linear_unpruned.weight)
            linear_pruned.bias_orig.copy_(linear_unpruned.bias)

    def get_final_reprs(self, s1, s1len, s2, s2len):
        s1enc = self.encoder(s1, s1len)
        s2enc = self.encoder(s2, s2len)

        diffs = s1enc - s2enc
        prods = s1enc * s2enc

        mlp_input = torch.cat([s1enc, s2enc, diffs, prods], 1)

        mlp_input = self.bn(mlp_input)
        mlp_input = self.dropout(mlp_input)


        rep = self.mlp[:-1](mlp_input)

        return rep

    def forward_from_final(self, rep):
        preds = self.mlp[-1:](rep)
        return preds

    def get_encoder(self):
        return self.encoder



class TextEncoder(nn.Module):
    def __init__(
        self, vocab_size, embedding_dim=300, hidden_dim=512, bidirectional=False
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.bidirectional = bidirectional
        self.emb = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=1)
        self.rnn = nn.LSTM(
            self.embedding_dim, self.hidden_dim, bidirectional=bidirectional
        )
        self.output_dim = self.hidden_dim

    def forward(self, s, slen):
        semb = self.emb(s)
        spk = pack_padded_sequence(semb, slen.cpu(), enforce_sorted=False)
        _, (hidden, cell) = self.rnn(spk)

        #retunr get all cell states w a param for the cell state #
        return hidden[-1]


    def get_states(self, s, slen):
        semb = self.emb(s)
        spk = pack_padded_sequence(semb, slen.cpu(), enforce_sorted=False)
        outputs, _ = self.rnn(spk)
        print(outputs)
        outputs_pad = pad_packed_sequence(outputs)[0]
        return outputs_pad #padded hidden states for each word

    def get_last_cell_state(self, s,slen):
        semb = self.emb(s)
        spk = pack_padded_sequence(semb, slen.cpu(), enforce_sorted=False)
        _, (hidden, cell) = self.rnn(spk)


        return cell[-1]


