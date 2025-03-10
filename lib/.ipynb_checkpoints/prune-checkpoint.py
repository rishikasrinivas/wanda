import time 
import heapq 
import torch 
import torch.nn as nn 
from .sparsegpt import SparseGPT 
from .layerwrapper import WrappedGPT
from .data import get_loaders 
from .collect_data import create_dataloaders
from .ablate import AblateGPT 
import collections
from  .model import BaseModel, Layer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
def find_layers(model, module, layers=[nn.Linear], name=''):
    
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            model, child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def find_final_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        if name == 'mlp' and name1 == '0':
            res.update(find_layers(
                child, layers=layers, name=name + '.' + name1 if name != '' else name1
            ))
    return res['mlp.0']


def check_sparsity(model, args):
    if args.model == 'bert':
        use_cache = model.config.use_cache 
        model.config.use_cache = False 
    if args.seg == 'bert':
        layers = model.encoder.encoder.layer
    elif args.seg == 'mlp':
        layers = model.mlp
    count = 0 
    total_params = 0
    for i, layer in enumerate(layers):
        subset = find_layers(model, layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()
        try:
            print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")
        except:
            pass

    if args.model == 'bert': 
        model.config.use_cache = use_cache 
    return float(count)/total_params 


def get_inputs_bert(model, embedder, dataloader, input_string, dtype, device):
    inps = torch.zeros((100, 100, 768), dtype=dtype, device=device)
    attention_mask = torch.zeros((100, 100, model.seqlen), dtype=dtype, device=device)
    inps.requires_grad = False
    for batch in dataloader:
        try:
            s1, s1len, s2, s2len, target = batch #s1 is longest sent x 100 since batch size is 100
            s1 = s1.to(device)
            s2 = s2.to(device)
            if input_string == 's1':
                s1_tokens = model.indices_to_bert_tokens(s1.transpose(1,0))
                s1_tokens = {k: v.to(device) for k, v in s1_tokens.items()}

                s1_tokens_embed = embedder(s1_tokens['input_ids'])

                inps[i][:s1_tokens_embed.shape[0], :]= s1_tokens_embed[:, 0, :]
                attention_mask[i][:s1_tokens['attention_mask'].shape[0], :] = s1_tokens['attention_mask']
            elif input_string == 's2':
                s2_tokens = model.indices_to_bert_tokens(s2.transpose(1,0))
                s2_tokens = {k: v.to(device) for k, v in s2_tokens.items()}
                s2_tokens = embedder(**s2_tokens)

                inps[i]= s2_tokens['input_ids']
                attention_mask[i] = s2_tokens['attention_mask']
            i+=1
        except ValueError:
            print("Caught ValueError")  
        outs = torch.zeros_like(inps)
        attention_mask = cache['attention_mask']
        position_ids = cache['position_ids']
    return inps, outs, None, attention_mask, position_ids

def get_inputs_bowman(model,seg, dataloader,input_string, dtype, device):
    inps = torch.zeros((100, 100, 300), dtype=dtype, device=device)
    lengths = torch.zeros((100, 100), dtype=dtype, device=device)
    inps.requires_grad = False
    i=0
    for batch in dataloader:
        try:
            s1, s1len, s2, s2len, target = batch #s1 is longest sent x 100 since batch size is 100
            s1 = s1.to(device)
            s2 = s2.to(device)
            
            if input_string == 's1':
                s1_enc= model.encoder.emb(s1)
                spk = pack_padded_sequence(s1_enc, s1len.cpu(), enforce_sorted=False)
                unpacked_tensor, lengths = nn.utils.rnn.pad_packed_sequence(
                    spk,
                    batch_first=True  # depending on your desired format
                )
                print(lengths.shape, unpacked_tensor.shape)
                inps[:,:unpacked_tensor.shape[1], :]=unpacked_tensor 
                lengths=lengths
                
            elif input_string == 's2':
                s2_enc= model.encoder.emb(s2)
                spk = pack_padded_sequence(s2_enc, s2len.cpu(), enforce_sorted=False)
                inps[i][:s2_enc.shape[0], :]= s2_enc
            i+=1
        except ValueError:
            print("Caught ValueError")  
        outs = torch.zeros_like(inps)
    return inps, lengths, outs
                        
def get_bert_encodings(model, embedder, s1, s2):
    s1_tokens = model.indices_to_bert_tokens(s1.transpose(1,0))
    s1_tokens = {k: v.to(device) for k, v in s1_tokens.items()}
    s1_tokens_embed = embedder(s1_tokens['input_ids'])

    s2_tokens = model.indices_to_bert_tokens(s2.transpose(1,0))
    s2_tokens = {k: v.to(device) for k, v in s2_tokens.items()}
    s2_tokens_embed = embedder(s2_tokens['input_ids'])

    s1enc= model.encoder(**s1_tokens)
    s1enc = s1enc.last_hidden_state[:, 0, :]
    s2enc= model.encoder(**s2_tokens)
    s2enc = s2enc.last_hidden_state[:, 0, :]
                        
    return s1enc, s2enc
                        
def get_inputs_mlp(model, embedder, args, dataloader, dtype, device):
    i=0
    in_feats = model.mlp[0].in_features
    out_feats = model.mlp[0].out_features
    inps = torch.zeros((100, 100, in_feats), dtype=dtype, device=device)
    inps.requires_grad = False
    for batch in dataloader:
        try:
            s1, s1len, s2, s2len, target = batch #s1 is longest sent x 100 since batch size is 100
            s1 = s1.to(device)
            s2 = s2.to(device)

            if args.model == 'bert':
                s1enc, s2enc = get_bert_encodings(model, embedder, s1,s2)
            elif args.model == 'bowman':
                s1enc = model.encoder(s1, s1len)
                s2enc = model.encoder(s2, s2len)

            diffs = s1enc - s2enc
            prods = s1enc * s2enc

            mlp_input = torch.cat([s1enc, s2enc, diffs, prods], 1)

            inps[i][:mlp_input.shape[0],:]= mlp_input
            i+=1

        except ValueError:
            print("Caught ValueError")  # Debug print '''
    outs = torch.zeros(100,100,out_feats)
    attention_mask = None
    position_ids = None
    return inps, outs, None, attention_mask, position_ids
                        
def prepare_calibration_input(model, args, dataloader, input_string, embedder, device):
    if args.model == 'bert':
        use_cache = model.config.use_cache
        model.config.use_cache = False
    

    dtype = next(iter(model.parameters())).dtype

    model = model.to(device)
    lengths=None
    
    # Add more debug
    print("Starting data loop")
    if args.seg == 'enc':
        if args.model == 'bert':
            inps, outs, attention_mask, position_ids = get_inputs_bert(model, embedder, dataloader, 's1', dtype, device)
        elif args.model == 'bowman':
            inps, lengths, outs = get_inputs_bowman(model,args.seg, dataloader, 's1', dtype, device) 
    else: #layer==mlp
        inps, outs, attention_mask, position_ids = get_inputs_mlp(model, embedder, args, dataloader, dtype, device)
    

    if args.model == 'bert':
        model.config.use_cache = use_cache

    return inps, outs, lengths, attention_mask, position_ids 

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity

def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    layers = model.model.layers 

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data 
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*args.sparsity_ratio)].cpu()
                W_mask = (W_metric<=thresh)

            W[W_mask] = 0
            
def get_first_linear_module(model):
    modules=collections.defaultdict(list)
    for name, module in model.named_modules():
        if name=="":
            continue
        layers=find_layers(model, module, layers=[nn.Embedding])
        if layers:
            return layers
        
def get_embedder(model):
    for name, module in model.encoder.named_modules():
        if name=='':
            continue
        return module
    
def get_modules(model):
    modules=collections.defaultdict(list)
    for name, module in model.named_modules:
        if name=="":
            continue
        print(name)
        modules[name]=module
        
    return modules

def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    if args.model == 'bert':
        use_cache = model.config.use_cache 
        model.config.use_cache = False 

    print("loading calibdation data")
    _,_,_, dataloaders=create_dataloaders(max_data=10000)
    dataloader = dataloaders['val']
    
    print("dataset loading complete")
    embedder = get_embedder(model)
    with torch.no_grad():
        inps, outs, lengths, attention_mask, position_ids = prepare_calibration_input(model, args, dataloader,'s1',embedder, device)

    layers = model.mlp #model.encoder.encoder.layer
    print(layers)
    for layer in layers: #[0] for mlp
        #get all the layers
        subset=find_layers(model, layer)
        print(layer)
        if layer == 'rnn':
            inps = nn.utils.rnn.pack_padded_sequence(
                inps,
                lengths.cpu(),  # lengths must be on CPU
                batch_first=True,  # if your tensor is [batch, seq, features]
                enforce_sorted=False  # set True if sequences are sorted by length
            )

        
        if not subset:
            continue
        
        
        #inps, outs, attention_mask, position_ids = inps.to(device), outs.to(device), attention_mask.to(device), position_ids.to(device)
        inps, outs  = inps.to(device), outs.to(device)
        
        #TODO; need to find either how to pass in the layer obj with the weight and prune or w\how to modify LAYER class
        wrapped_layers = {}
        for name in subset:
            wrapped_layers[subset[name]] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(name.register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
               
                input_tmp = inps[j].unsqueeze(0)
                outs[j] = layer(input_tmp)[0]
                    
        
           
        for h in handles:
            h.remove()
        #prunes eavh layer in the module
        #print(subset)
        for name in subset:
            print(f"pruning layer {type(layer)} name {name}, {subset[name]}")
            subset_value = subset[name]
            W_metric = torch.abs(subset_value.weight.data) * torch.sqrt(wrapped_layers[subset_value].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero 
            
        #passes the inps thru the lauer to get the inputs to thenext layer
        for j in range(args.nsamples):
            input_tmp = inps[j].unsqueeze(0)
            outs[j]= layer(input_tmp)[0]
              
        inps, outs = outs, inps
        if args.seg == 'mlp':
            break
    if args.model == 'bert':
        model.config.use_cache = use_cache 
    torch.cuda.empty_cache()
    torch.save(model.state_dict(), "Results/bert_prune_wanda.pth")

def prune_wanda_bert(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    (tokens, true_class)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device) #no need in boewman

    layers = model.model.layers
    #run this
    print("In prune: layers=", layers)
    
    layer = layers[i]
    print(f"In prune iter {i}: layers{i}=", layer)

    subset = find_layers(layer)
    print(f"In prune subset=", subset)


    if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
        dev = model.hf_device_map[f"model.layers.{i}"]
        inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

    wrapped_layers = {}
    for name in subset:
        wrapped_layers[name] = WrappedGPT(subset[name])


    def add_batch(name):
        def tmp(_, inp, out):
            wrapped_layers[name].add_batch(inp[0].data, out.data)
        return tmp

    handles = []
    for name in wrapped_layers:
        handles.append(subset[name].register_forward_hook(add_batch(name)))
    for j in range(args.nsamples):
        with torch.no_grad():
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
    for h in handles:
        h.remove()

    for name in subset:
        print(f"pruning layer {i} name {name}")
        W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

        W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
        if prune_n != 0:
            # structured n:m sparsity
            for ii in range(W_metric.shape[1]):
                if ii % prune_m == 0:
                    tmp = W_metric[:,ii:(ii+prune_m)].float()
                    W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
        else:
            sort_res = torch.sort(W_metric, dim=-1, stable=True)

            if args.use_variant:
                # wanda variant 
                tmp_metric = torch.cumsum(sort_res[0], dim=1)
                sum_before = W_metric.sum(dim=1)

                alpha = 0.4
                alpha_hist = [0., 0.8]
                W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                    if cur_sparsity > args.sparsity_ratio:
                        alpha_new = (alpha + alpha_hist[0]) / 2.0
                        alpha_hist[1] = alpha
                    else:
                        alpha_new = (alpha + alpha_hist[1]) / 2.0
                        alpha_hist[0] = alpha

                    alpha = alpha_new 
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
            else:
                # unstructured pruning
                indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                W_mask.scatter_(1, indices, True)

        subset[name].weight.data[W_mask] = 0  ## set weights to zero 

    for j in range(args.nsamples):
        with torch.no_grad():
            for layer_ in layer:
                outs[j] = layer_(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
    inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()


@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()



@torch.no_grad()
def prune_ablate(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = AblateGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            if args.prune_method == "ablate_wanda_seq":
                prune_mask = gpts[name].get_wanda_mask(args.sparsity_ratio, prune_n, prune_m)
            elif args.prune_method == "ablate_mag_seq":
                prune_mask = gpts[name].get_mag_mask(args.sparsity_ratio, prune_n, prune_m)
            elif "iter" in args.prune_method:
                prune_mask = None 

            gpts[name].fasterprune(args, args.sparsity_ratio, mask=prune_mask, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()