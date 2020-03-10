import os
import json
import glob
import argparse
import typing

import pandas as pd
import numpy as np
import torch
from transformers import *
import tqdm
import h5py
from nltk.tokenize import sent_tokenize, word_tokenize


#  Key | Model | Tokenizer | Pretrained weights shortcut
MODELS = {
    "bert": (BertModel, BertTokenizer, 'bert-base-uncased'),
    "gpt": (OpenAIGPTModel, OpenAIGPTTokenizer, 'openai-gpt'),
    "gpt2": (GPT2Model, GPT2Tokenizer, 'gpt2'),
    "ctrl": (CTRLModel, CTRLTokenizer, 'ctrl'),
    "transformerxl": (TransfoXLModel, TransfoXLTokenizer, 'transfo-xl-wt103'),
    "xlnet": (XLNetModel, XLNetTokenizer, 'xlnet-base-cased'),
    "xlm": (XLMModel, XLMTokenizer, 'xlm-mlm-enfr-1024'),
    "distilbert": (DistilBertModel, DistilBertTokenizer, 'distilbert-base-cased'),
    "roberta": (RobertaModel, RobertaTokenizer, 'roberta-base'),
    "xlmroberta": (XLMRobertaModel, XLMRobertaTokenizer, 'xlm-roberta-base'),
}


def make_hdf5_file(sentence_to_index: typing.Dict[str, str], vectors: typing.Dict[str, np.array], output_file_path: str):
    """
    Makes the hdf5 file needed to train the probes.

    Arguments:
        sentence_to_index: a dictionary mapping a sentence to an index (in string form, aka "1")
        vectors: a dictionary mapping a string number (aka "1") to a numpy array representing the embedding
        output_file_path: the output location to write the hdf5 file
    """
    with h5py.File(output_file_path, 'w') as fout:
        for key, embeddings in vectors.items():
            fout.create_dataset(
                str(key),
                embeddings.shape, dtype='float32',
                data=embeddings)
        sentence_index_dataset = fout.create_dataset(
            "sentence_to_index",
            (1,),
            dtype=h5py.special_dtype(vlen=str))
        sentence_index_dataset[0] = json.dumps(sentence_to_index)


def raw_line_count(filename: str):
    """ 
    A quick helper function to get the total line count of the file.
    Should be fast, but feel free to take it out if not
    Taken from: https://stackoverflow.com/questions/845058/how-to-get-line-count-of-a-large-file-cheaply-in-python
    """
    f = open(filename, 'rb')
    lines = 0
    buf_size = 1024 * 1024
    read_f = f.raw.read

    buf = read_f(buf_size)
    while buf:
        lines += buf.count(b'\n')
        buf = read_f(buf_size)

    return lines


def load_model_and_write_hdf5(args: argparse.Namespace):
    """
    Loads a pretrained huggingface model and gets the embeddings for each sentence in the file of `args.data_path`
    Writes this into an hdf5 file to `args.output_path`
    """
    print("Loading model and tokenizer")
    model_details = MODELS[args.model_name]
    weight_path = model_details[2] if args.model_weights is None else args.model_weights
    model = model_details[0].from_pretrained(weight_path, output_hidden_states=True)
    tokenizer = model_details[1].from_pretrained(weight_path)

    if args.cuda:
        model = model.cuda()

    sentence_to_index = {}
    index_to_vector = {}

    line_count = raw_line_count(args.data_path) # for progress bar

    print("Embedding data file...")
    encoded_special_tokens = tokenizer.encode(tokenizer.all_special_tokens, add_special_tokens=False)
    with open(args.data_path, "r") as fin:
        for index, sentence in enumerate(tqdm.tqdm(fin, total=line_count)):
            token_to_subword_mapping = {}
            clean_sent = sentence.strip() # remove \n
            sentence_to_index[clean_sent] = str(index)
            # create mapping of token -> ids
            map_index = 0
            sentence_tokens = clean_sent.split(" ")
            for token_index, token in enumerate(sentence_tokens):
                tokenized_len = len(tokenizer.encode(token, add_special_tokens=False))
                if tokenized_len == 0:
                    import pdb; pdb.set_trace()
                    raise(Exception("Zero length token"))
                # map the index of the token to the index the token will span in the wordpiece tokens
                token_to_subword_mapping[token_index] = list(range(map_index, map_index + tokenized_len))
                map_index += tokenized_len

            encoded_sent = tokenizer.encode(clean_sent, add_special_tokens=True)
            keep_mask = [1 if token not in encoded_special_tokens else 0 for token in encoded_sent] # to subset output by
            input_ids = torch.tensor([encoded_sent])  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
            with torch.no_grad():
                if args.cuda:
                    input_ids = input_ids.cuda()
                last_hidden_state, pooled_output, all_layers  = model(input_ids)
                embedding_output, layer_output = all_layers[0], all_layers[1:]
                if args.all_layers:
                    representation = torch.cat(list(layer_output), dim=0)
                else:
                    representation = last_hidden_state

            # use mapping of token -> ids to sum up word_pieces and discard [sep][cls] tokens
            # get the indexes corresponding to each token.  Sum the wordpiece tokens and then unsqueeze to get (1, hidden_dim) for each token, then concat together
            no_special_tokens = representation[:, torch.tensor(keep_mask).nonzero().squeeze(-1)]
            # vectors shape (seq_len, num_layers, dim_embedding)
            vectors = torch.cat([no_special_tokens.index_select(1, torch.tensor(values).cuda()).sum(dim=1).unsqueeze(0) for values in token_to_subword_mapping.values()], dim=0)
            vectors = vectors.permute(1, 0, 2) # shape (n_layers, seq_len, dim_embedding)
            assert vectors.shape[1] == len(sentence_tokens), "error in shapes"
            index_to_vector[str(index)] = vectors.detach().cpu().numpy()

    print("Writing saved vectors as an hdf5 file...")
    print("Shapes of the written vectors are:", vectors.shape)
    make_hdf5_file(sentence_to_index, index_to_vector, args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="the model you want to use to generate the embedding")
    parser.add_argument("data_path", help="the path to the data you want to embed")
    parser.add_argument("output_path", help="the path you want to use to store the hdf5 file")
    parser.add_argument("--model_weights", help="the model weights you want to use, loaded from file", default=None)
    parser.add_argument("--cuda", action="store_true", help="use cuda to speed up the embedding process", default=False)
    parser.add_argument("--all_layers", action="store_true", help="store each layer, instead of just the last", default=False)
    args = parser.parse_args()
    load_model_and_write_hdf5(args)