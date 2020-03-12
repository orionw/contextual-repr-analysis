import os
import json
import glob
import argparse
import typing
import random
import time
from collections import Counter, OrderedDict

import pandas as pd
import numpy as np
import torch
from transformers import *
import tqdm
import h5py
from nltk.tokenize import sent_tokenize, word_tokenize

from scripts.huggingface_extensions import RobertaForSequenceClassificationGLUE


#  Key | Model | Tokenizer | Pretrained weights shortcut

MODELS = {
    "bert": (BertConfig, BertModel, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetModel, XLNetTokenizer),
    "xlm": (XLMConfig, XLMModel, XLMTokenizer),
    "roberta": (
        RobertaConfig,
        RobertaForSequenceClassificationGLUE,
        RobertaTokenizer,
    ),
    "distilbert": (
        DistilBertConfig,
        DistilBertModel,
        DistilBertTokenizer,
    ),
    "albert": (AlbertConfig, AlbertModel, AlbertTokenizer),
    "xlmroberta": (
        XLMRobertaConfig,
        XLMRobertaModel,
        XLMRobertaTokenizer,
    ),
    "flaubert": (
        FlaubertConfig,
        FlaubertModel,
        FlaubertTokenizer,
    ),
    "gpt": (OpenAIGPTConfig, OpenAIGPTModel, OpenAIGPTTokenizer),
    "gpt2": (GPT2Config, GPT2Model, GPT2Tokenizer),
    "ctrl": (CTRLConfig, CTRLModel, CTRLTokenizer),
    "transformerxl": (TransfoXLConfig, TransfoXLModel, TransfoXLTokenizer),
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



def get_subword_mapping(token_to_subword_mapping: typing.Dict, clean_sent: str, tokenizer):
    """ A function to compute the mapping from word tokens to subword tokens for later usage """
    map_index = 0
    sentence_tokens = clean_sent.split(" ")
    for token_index, token in enumerate(sentence_tokens):
        tokenized_len = len(tokenizer.encode(token, add_special_tokens=False))
        if tokenized_len == 0:
            import pdb; pdb.set_trace()
            raise(Exception("Zero length token"))
        # incase the individual mapping does not align to the combined mapping, slower but accurate
        tokens_til_now = tokenizer.encode(" ".join(sentence_tokens[:token_index]), add_special_tokens=False)
        all_tokens_to_now = tokenizer.encode(" ".join(sentence_tokens[:token_index+1]), add_special_tokens=False)
        new_tokens = all_tokens_to_now[-(len(all_tokens_to_now) - len(tokens_til_now)):]
        # map the index of the token to the index the token will span in the wordpiece tokens
        token_to_subword_mapping[token_index] = list(range(map_index, map_index + len(new_tokens)))
        map_index += len(new_tokens)
    return token_to_subword_mapping, sentence_tokens



def load_model_and_write_hdf5(args: argparse.Namespace):
    """
    Loads a pretrained huggingface model and gets the embeddings for each sentence in the file of `args.data_path`
    Writes this into an hdf5 file to `args.output_path`
    """
    print("Loading model and tokenizer")
    if args.model_name_or_path is None:
        args.model_name_or_path = args.model_class # no weights given, use default
    config, model_class, tokenizer_class = MODELS[args.model_class]
    model = model_class.from_pretrained(args.model_name_or_path, output_hidden_states=True)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

    if args.cuda:
        model = model.cuda()

    sentence_to_index = {}
    index_to_vector = {}

    line_count = raw_line_count(args.data_path) # for progress bar

    print("Mapping tokens from the data file...")
    encoded_special_tokens = tokenizer.encode(tokenizer.all_special_tokens, add_special_tokens=False)
    with open(args.data_path, "r") as fin:
        full_list = []
        len_list = []
        for index, sentence in enumerate(tqdm.tqdm(fin, total=line_count)):
            token_to_subword_mapping = {}
            clean_sent = sentence.strip() # remove \n
            sentence_to_index[clean_sent] = str(index)
            # create mapping of token -> ids
            token_to_subword_mapping, sentence_tokens = get_subword_mapping(token_to_subword_mapping, clean_sent, tokenizer)
            encoded_sent = tokenizer.encode(clean_sent, add_special_tokens=True) # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
            keep_mask = [1 if token not in encoded_special_tokens else 0 for token in encoded_sent] # to subset output by
            full_list.append({
                "index": index,
                "keep_mask": keep_mask,
                "encoded_sent": encoded_sent,
                "token_to_subword": token_to_subword_mapping,
                "sentence_len": len(sentence_tokens),
                "sentence": clean_sent
            })
            len_list.append(len(encoded_sent))

    # to save time, batch with the sequence lengths
    freq_sent_len = OrderedDict(sorted(Counter(len_list).items())) # sorted from smaller len to greatest
    list_of_sents = sorted(full_list, key=lambda x: len(x['encoded_sent']))
    print("About to process seq_lens and batch_sizes of", freq_sent_len)

    cur_index = 0
    print("Batching vector representations...")
    for len_of_sent, count in tqdm.tqdm(freq_sent_len.items()):
        batch = list_of_sents[cur_index:cur_index+count]
        batch_size = count

        # TODO max the batch count, for size constraint reasons

        # shapes (batch_size, seq_len)
        input_ids = torch.cat([torch.tensor(item["encoded_sent"]).unsqueeze(0).cuda() if args.cuda \
                                else torch.tensor([item["encoded_sent"]]).unsqueeze(0) for item in batch])
        keep_masks = torch.cat([torch.tensor(item["keep_mask"]).unsqueeze(0).cuda() if args.cuda else \
                                 torch.tensor(item["keep_mask"]).unsqueeze(0) for item in batch])
        # unpack
        token_to_subword_mappings = [item["token_to_subword"] for item in batch]
        original_indexes = [item["index"] for item in batch]
        sentence_lengths = [item["sentence_len"] for item in batch]
        sentences = [item["sentence"] for item in batch]

        assert (keep_masks[0]==keep_masks).all(), "different keep masks in batch, uh-oh, they should all be the same"
        with torch.no_grad():
            last_hidden_state, _, all_layers  = model(input_ids) # (batch_size, seq_len, embed_dim)
            embedding_output, layer_output = all_layers[0], all_layers[1:]
            if args.all_layers:
                layer_output = [layer.unsqueeze(0) for layer in layer_output] # add a dim for layer num
                representation = torch.cat(list(layer_output), dim=0) # combine layers to one
            else:
                representation = last_hidden_state.unsqueeze(0)

        # use mapping of token -> ids to sum up word_pieces and discard [sep][cls] tokens
        # get the indexes corresponding to each token.  Sum the wordpiece tokens and then unsqueeze to get (1, hidden_dim) for each token, then concat together
        no_special_tokens = representation[:, :, keep_masks[0, :].nonzero().squeeze(-1)] # shape (n_layers, batch_size, seq_len, embed_dim)
        for sent_num in range(batch_size): # range batch size
            cur_token_to_subword_mapping = token_to_subword_mappings[sent_num]
            cur_vec = no_special_tokens[:, sent_num, :, :] # get only the vector to map sub-words, shape (n_layers, seq_len, embed_dim)
            # take the tokens from the seq_len dim, dim=1 and sum them so it can concat sub-word tokens if needed
            vector = torch.cat([cur_vec.index_select(1, torch.tensor(values).cuda()).sum(dim=1).unsqueeze(0) for values in cur_token_to_subword_mapping.values()], dim=0)
            vector = vector.permute(1, 0, 2) # shape (n_layers, seq_len, dim_embedding)
            assert vector.shape[1] == sentence_lengths[sent_num], "error in shapes"
            index_to_vector[str(original_indexes[sent_num])] = vector.detach().cpu().numpy()
        cur_index += count

    print("Writing saved vectors as an hdf5 file to {}...".format(args.output_path))
    print("Shapes of the written vectors are:", vector.shape)
    repeats = 5
    while (repeats > 0):
        try:
            make_hdf5_file(sentence_to_index, index_to_vector, args.output_path)
            repeats = 0
        except Exception as e:
            time.sleep(random.randint(10, 40))
            repeats -= 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_class", help="the model class you want to use to generate the embedding")
    parser.add_argument("data_path", help="the path to the data you want to embed")
    parser.add_argument("output_path", help="the path you want to use to store the hdf5 file")
    parser.add_argument("--model_name_or_path", help="the specific model you want to use to generate the embedding", default=None)
    parser.add_argument("--cuda", action="store_true", help="use cuda to speed up the embedding process", default=False)
    parser.add_argument("--all_layers", action="store_true", help="store each layer, instead of just the last", default=False)
    args = parser.parse_args()
    load_model_and_write_hdf5(args)