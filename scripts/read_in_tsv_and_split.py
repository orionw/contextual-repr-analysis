import os
import pandas as pd
from allennlp.data.dataset_readers import UniversalDependenciesDatasetReader
from allennlp.data.iterators import BucketIterator
import numpy as np
import tqdm
import json

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def split_event_factuality(path_to_tsv: str):
    event = pd.read_csv(os.path.join(path_to_tsv, "it-happened_eng_ud1.2_07092017.tsv"), sep='\t', header=0, keep_default_na=False, na_values=['', "nan", "na"])
    event["Confidence"] = pd.to_numeric(event["Confidence"], errors='coerce')
    train = event[event["Split"] == "train"]
    dev = event[event["Split"] == "dev"]
    test = event[event["Split"] == "test"]

    reader = UniversalDependenciesDatasetReader()
    for split, data_split in zip(["train", "dev", "test"], [train, dev, test]):
        dataset = reader.read(os.path.join(path_to_tsv, "en-ud-{}.conllu".format(split)))
        new_dataset = {}
        loop = tqdm.tqdm(dataset)
        for index, instance in enumerate(loop):
            raw_tokens = [word.__dict__["text"] for word in dataset[index]["words"].__dict__["tokens"]]
            sentence = " ".join(raw_tokens)
            relevant_annotations = data_split[data_split["Sentence.ID"] == "en-ud-{}.conllu {}".format(split, index + 1)]
            for name, group_df in relevant_annotations.groupby("Pred.Token"):
                happened = int((group_df["Happened"] == True).all())
                if happened == 0:
                    happened = -1
                nan_mean_of_group = np.nanmean(group_df["Confidence"])
                if np.isnan(nan_mean_of_group):
                    continue
                ave_score = nan_mean_of_group * 3 / 4 * happened # account for the [-3, 3] scale
                predicate_position = [group_df["Pred.Token"].iloc[0]]
                assert len(predicate_position) == 1, "got more than one predicate position"
                num_positions = len(set(group_df["Pred.Token"].to_list()))
                try:
                    assert num_positions == 1, "got diff than one position, got {}".format(num_positions)
                except Exception as e:
                    print(e)
                    import pdb; pdb.set_trace()
                new_dataset[str(index)] = {
                    "predicate_indices": predicate_position,
                    "labels": ave_score,
                    "sentence": raw_tokens,
                    "raw_tokens": raw_tokens # not sure the diff here, if needed
                }
            loop.update(1)
        with open(os.path.join(path_to_tsv, "it-happened_eng_ud1.2_07092017.{}.json".format(split)), "w") as fout:
            json.dump(new_dataset, fout, cls=MyEncoder)

    print("Done!")


if __name__ == "__main__":
    split_event_factuality(os.path.join("..", "data", "event_factuality"))