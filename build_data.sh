set -eu

if [ ! -e data ]; then
    mkdir data
fi

cd data
wget -O JSNLI.zip https://nlp.ist.i.kyoto-u.ac.jp/DLcounter/lime.cgi\?down\=https://nlp.ist.i.kyoto-u.ac.jp/nl-resource/JSNLI/jsnli_1.1.zip\&name\=JSNLI.zip
unzip JSNLI.zip

# Convert to jsonl format
cat jsnli_1.1/train_w_filtering.tsv | python ../convert_data_format.py >train_orig.jsonl
cat jsnli_1.1/dev.tsv | python ../convert_data_format.py >test.jsonl

# Split train/val data
cat train_orig.jsonl | head -n 523005 >train.jsonl
cat train_orig.jsonl | tail -n 10000 >val.jsonl