import argparse
import os 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version
import lib.model as models
from lib.prune import prune_wanda, prune_magnitude, prune_sparsegpt, prune_ablate, check_sparsity, find_layers
from lib.eval import eval_ppl, eval_zero_shot

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())
from torch.utils.data import DataLoader

from data.snli import SNLI, pad_collate
def create_dataloaders(max_data):
    root_dir="../CCE_NLI/models/DataLoaders"
    if not ('train_dataset.pth' in os.listdir(root_dir) and 'val_dataset.pth' in os.listdir(root_dir) and 'test_dataset.pth' in os.listdir(root_dir)):
        train = SNLI("data/snli_1.0", "train", max_data=max_data)
        train_loader = DataLoader(
            train,
            batch_size=100,
            shuffle=True,
            pin_memory=False,
            num_workers=0,
            collate_fn=pad_collate,
        )
        torch.save(train_loader.dataset, f'{root_dir}/train_dataset.pth')
        
        val = SNLI("data/snli_1.0","dev",max_data=max_data,vocab=(train.stoi, train.itos),unknowns=False)
        val_loader = DataLoader(
            val, 
            batch_size=100, 
            shuffle=False,
            pin_memory=True, 
            num_workers=0, 
            collate_fn=pad_collate
        
        )
        torch.save(val_loader.dataset, f'{root_dir}/val_dataset.pth')
        
        test = SNLI("data/snli_1.0", "test", max_data=max_data, vocab=(train.stoi, train.itos), unknowns=True)
        test_loader = DataLoader(
            test,
            batch_size=100,
            shuffle=False,
            pin_memory=True,
            num_workers=0,
            collate_fn=pad_collate,
        )
        torch.save(test_loader.dataset, f'{root_dir}/test_dataset.pth')
    else:
        train_dataset = torch.load(f'{root_dir}/train_dataset.pth')
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=100, 
            shuffle=True, 
            pin_memory=False,
            num_workers=0,
            collate_fn=pad_collate
        )
        
        val_dataset = torch.load(f'{root_dir}/val_dataset.pth')
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=100, 
            shuffle=False, 
            pin_memory=True, 
            num_workers=0, 
            collate_fn=pad_collate
        )
        
        test_dataset = torch.load(f'{root_dir}/test_dataset.pth')
        test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=100, 
            shuffle=False,
            pin_memory=True,
            num_workers=0,
            collate_fn=pad_collate
        )
        
        
    
    dataloaders = {
        'train': train_loader,
        'val':val_loader,
        'test': test_loader
    }
    return train_loader.dataset, val_loader.dataset,test_loader.dataset, dataloaders

def get_llm(model_name, cache_dir="llm_weights"):
    train,val,test,dataloaders=create_dataloaders(max_data=10000)
    enc = models.TextEncoder(
        vocab_size=len(train.stoi), embedding_dim=300, hidden_dim=512
    )
    model = models.BertEntailmentClassifier(vocab={'stoi': train.stoi, 'itos': train.itos})
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=["bowman", "bert"])
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, choices=["magnitude", "wanda", "sparsegpt", 
                        "ablate_mag_seq", "ablate_wanda_seq", "ablate_mag_iter", "ablate_wanda_iter", "search"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')

    parser.add_argument("--eval_zero_shot", action="store_true")
    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=False)

    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)

    if args.sparsity_ratio != 0:
        print("pruning starts")
        if args.prune_method == "wanda":
            prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "magnitude":
            prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "sparsegpt":
            prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif "ablate" in args.prune_method:
            prune_ablate(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

    ################################################################
    print("*"*30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*"*30)
    ################################################################
    ppl_test = eval_ppl(args, model, tokenizer, device)
    print(f"wikitext perplexity {ppl_test}")

    if not os.path.exists(args.save):
        os.makedirs(args.save)
    save_filepath = os.path.join(args.save, f"log_{args.prune_method}.txt")
    with open(save_filepath, "w") as f:
        print("method\tactual_sparsity\tppl_test", file=f, flush=True)
        print(f"{args.prune_method}\t{sparsity_ratio:.4f}\t{ppl_test:.4f}", file=f, flush=True)

    if args.eval_zero_shot:
        accelerate=False
        if "30b" in args.model or "65b" in args.model or "70b" in args.model:
            accelerate=True

        task_list = ["boolq", "rte","hellaswag","winogrande", "arc_easy","arc_challenge", "openbookqa"]
        num_shot = 0
        results = eval_zero_shot(args.model, model, tokenizer, task_list, num_shot, accelerate)
        print("********************************")
        print("zero_shot evaluation results")
        print(results)

    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)

if __name__ == '__main__':
    main()