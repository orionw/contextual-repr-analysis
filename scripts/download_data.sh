# This script downloads all of the publically available data into the data folders
cd data/

## POS Tagging
# (PTB) conll-x
cd pos/
wget http://fileadmin.cs.lth.se/nlp/software/pennconverter/pennconverter.jar

for file in path/to/mrg/files/*/*.mrg;
 do 
 # this is not fast, but it does seem to work
 java -jar pennconverter.jar < $file >  all/$(basename $file).conllx
done

# make splits
cat all/wsj_{00,01,02,03,04,05,06,07,08,09,10,11,12,13,14,15,16,17,18}*  > wsj.train.conllx
cat all/wsj_{19,20,21}*  > wsj.dev.conllx
cat all/wsj_{22,23,24}*  > wsj.test.conllx

# python3 scripts/get_config_sentences.py --config-paths ./experiment_configs/bert_base_cased/ptb_pos_tagging.json --output-path ./data/pos/sentences_ptb.txt
# python3 create_hdf5_file_from_huggingface_model.py bert ../data/pos/sentences_ptb.txt ../contextualizers/bert_base_cased/ptb_pos.hdf5 --cuda
# allennlp train experiment_configs/bert_base_cased/ptb_pos_tagging.json -s ptb_train --include-package contexteval

cd ../

# EWT POS
cd pos/
wget https://github.com/UniversalDependencies/UD_English-EWT/raw/master/en_ewt-ud-dev.conllu
wget https://github.com/UniversalDependencies/UD_English-EWT/raw/master/en_ewt-ud-train.conllu
wget https://github.com/UniversalDependencies/UD_English-EWT/raw/master/en_ewt-ud-test.conllu
cd ..

# python3 scripts/get_config_sentences.py --config-paths ./experiment_configs/bert_base_cased/ewt_pos_tagging.json --output-path ./data/pos/sentences_ewt.txt
# python3 create_hdf5_file_from_huggingface_model.py bert ../data/pos/sentences_ewt.txt ../contextualizers/bert_base_cased/ewt_pos.hdf5 --cuda


# CCG supertagging (CCG)
# form the PTB, LDC
# TODO

# semantic tagging task (ST)
# dataset of Bjerva et al. 2016
cd semantic_tagging/
git clone https://github.com/ginesam/semtagger.git
cd semtagger
cd data/
echo 'export DIR_ROOT=$PWD' >> to_add.txt
echo 'set -a' >> to_add.txt
echo 'source config.sh' >> to_add.txt
echo 'set +a' >> to_add.txt
sed -i -e '2 { r to_add.txt' -e 'N; }' prepare_data.sh 
awk 'NR==95{print "exit 1"}1' prepare_data.sh 

cd ../
bash data/prepare_data.sh
cp data/pmb/en/gold/test.gold ../semtag_test.conll
cp data/pmb/en/gold/train.gold ../semtag_train.conll
cp data/pmb/en/gold/train.gold ../semtag_dev.conll
cd ../../

# python3 scripts/get_config_sentences.py --config-paths ./experiment_configs/bert_base_cased/semantic_tagging.json --output-path ./data/semantic_tagging/sentences.txt
# python3 create_hdf5_file_from_huggingface_model.py bert ../data/semantic_tagging/sentences.txt ../contextualizers/bert_base_cased/semantic_tagging.hdf5 --cuda
# allennlp train experiment_configs/bert_base_cased/semantic_tagging.json -s st_train     --include-package contexteval

# Preposition supersense disambiguation (PS-fxn and PS-role), adaposition
# Use the STREUSLE 4.0 corpus
cd adposition_supersenses/
wget https://github.com/nert-nlp/streusle/raw/master/test/streusle.ud_test.json
wget https://github.com/nert-nlp/streusle/raw/master/train/streusle.ud_train.json
wget https://github.com/nert-nlp/streusle/raw/master/dev/streusle.ud_dev.json
cd ../

# python3 scripts/get_config_sentences.py --config-paths experiment_configs/bert_base_cased/adposition_supersense_tagging_function.json --output-path ./data/adposition_supersenses/sentences.txt
# cd scripts/
# python3 create_hdf5_file_from_huggingface_model.py bert ../data/adposition_supersenses/sentences.txt ../contextualizers/bert_base_cased/adposition_supersense_tagging.hdf5 --cuda
# cd ../
# allennlp train experiment_configs/bert_base_cased/adposition_supersense_tagging_function.json -s adposition_func_train --include-package contexteval
# allennlp train experiment_configs/bert_base_cased/adposition_supersense_tagging_role.json -s adposition_role_train --include-package contexteval

# event factuality (EF)
cd event_factuality/
wget http://decomp.io/projects/factuality/factuality_eng_udewt.tar.gz
tar -xvf factuality_eng_udewt.tar.gz 
cp it-happened/it-happened_eng_ud1.2_07092017.tsv .
wget https://github.com/UniversalDependencies/UD_English-EWT/raw/17a19823d04bf078e8a9be532118fe0ff1011934/en-ud-dev.conllu
wget https://github.com/UniversalDependencies/UD_English-EWT/raw/17a19823d04bf078e8a9be532118fe0ff1011934/en-ud-train.conllu
wget https://github.com/UniversalDependencies/UD_English-EWT/raw/17a19823d04bf078e8a9be532118fe0ff1011934/en-ud-test.conllu
cd ../scripts
python3 read_in_tsv_and_split.py
cd ../

# python3 scripts/get_config_sentences.py --config-paths ./experiment_configs/bert_base_cased/event_factuality.json --output-path ./data/event_factuality/sentences.txt
# cd scripts/
# python3 create_hdf5_file_from_huggingface_model.py bert ../data/event_factuality/sentences.txt ../contextualizers/bert_base_cased/event_factuality.hdf5 --cuda
# cd ../
# allennlp train experiment_configs/bert_base_cased/event_factuality.json -s factuality_train     --include-package contexteval
# TODO: numbers seem a little low, something off?


## Syntactic Chunking data (Chunk)
cd chunking/
wget http://www.clips.uantwerpen.be/conll2000/chunking/train.txt.gz
wget http://www.clips.uantwerpen.be/conll2000/chunking/test.txt.gz
gunzip test.txt.gz
gunzip train.txt.gz
cp train.txt eng_chunking_train.conll
mv train.txt eng_chunking_dev.conll
mv test.txt eng_chunking_test.conll
cd ..

# python3 scripts/get_config_sentences.py --config-paths ./experiment_configs/bert_base_cased/conll2000_chunking.json --output-path ./data/chunking/sentences.txt
# python3 create_hdf5_file_from_huggingface_model.py bert ../data/semantic_tagging/sentences.txt ../c

# Named entity recognition (NER)
cd ner/
wget https://github.com/rishiabhishek/Important-Datasets/raw/master/conll2003/eng.testa
wget https://github.com/rishiabhishek/Important-Datasets/raw/master/conll2003/eng.testb
wget https://github.com/rishiabhishek/Important-Datasets/raw/master/conll2003/eng.train
cd ..

# python3 scripts/get_config_sentences.py --config-paths ./experiment_configs/bert_base_cased/conll2003_ner.json --output-path ./data/ner/sentences.txt
# python3 create_hdf5_file_from_huggingface_model.py bert ../data/ner/sentences.txt ../contextualizers/bert_base_cased/conll2003_ner.hdf5 --cuda
# allennlp train experiment_configs/bert_base_cased/conll2003_ner.json -s ner_train     --include-package contexteval

# Grammatical error detection (GED)
# From the First Certificate in English
cd grammatical_error_correction/
wget https://s3-eu-west-1.amazonaws.com/ilexir-website-media/fce-error-detection.tar.gz
tar -xvzf fce-error-detection.tar.gz
mv fce-error-detection/tsv/* .
mv fce-public.dev.original.tsv fce-public.dev
mv fce-public.train.original.tsv fce-public.train
mv fce-public.test.original.tsv fce-public.test
cd ../

# python3 scripts/get_config_sentences.py --config-paths ./experiment_configs/bert_base_cased/grammatical_error_correction.json --output-path ./data/grammatical_error_correction/sentences.txt
# python3 create_hdf5_file_from_huggingface_model.py bert ../data/grammatical_error_correction/sentences.txt ../contextualizers/bert_base_cased/grammatical_error_correction.hdf5 --cuda
# allennlp train experiment_configs/bert_base_cased/grammatical_error_correction.json -s grammar_train --include-package contexteval 



# Coordinate boundary / conjunct identification (Conj)
cd coordination_boundary/
git clone https://github.com/Jess1ca/CoordinationExtPTB.git
# need to collapse ptb into one folder with all wsj files from the parsed mrg files
cd CoordinationExtPTB
python2 run.py ../path/to/parsed/mrg/wsj/files
cat PTB.ext | wc -l # should be 49208
head -n +34446 PTB.ext > ../conjunct_id.train # TODO: fix this to grab the correct file sets
head -n +34446 PTB.ext > PTB.exe
head -n +7381 PTB.ext > ../conjunct_id.dev
tail -n +7381 PTB.ext > ../conjunct_id.test

# TODO: how to split this?
python3 scripts/get_config_sentences.py --config-paths experiment_configs/bert_base_cased/conjunct_identification.json --output-path ./data/coordination_boundary/sentences.txt
# cd scripts/
# python3 create_hdf5_file_from_huggingface_model.py bert ../data/adposition_supersenses/sentences.txt ../contextualizers/bert_base_cased/adposition_supersense_tagging.hdf5 --cuda
# cd ../
# allennlp train experiment_configs/bert_base_cased/adposition_supersense_tagging_function.json -s adposition_func_train --include-package contexteval
# allennlp train experiment_configs/bert_base_cased/adposition_supersense_tagging_role.json -s adposition_role_train --include-package contexteval

cd ../





# the following need negative eamples generated
# TODO

# Syntactic Dependency Arc Prediction
# We use the PTB (converted to UD) and the UD-EWT
# TODO


## Semantic Dependency Parsing
cd semantic_dependency/
curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-1956{/osdp-12.tgz}
cd ..


# Coreference Arc Prediction
https://conll.cemantix.org/2012/data.html
# TODO

# syntactic constituency ancestor tagging (GGParent)
# not sure where this came from, just says PTB





