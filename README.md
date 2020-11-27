## Commands

- Preprocess

  ```
  python3 -m model.long_summarization.scripts.json_to_bin --infile ./dataset/long_summarization_data/arxiv-dataset/train.txt --outfile ./dataset/long_summarization_data/recursive-arxiv-dataset/output/train.bin --vocab_file ./dataset/long_summarization_data/recursive-arxiv-dataset/output/vocab
  
  python3 -m model.long_summarization.scripts.json_to_bin --infile ./dataset/long_summarization_data/arxiv-dataset/val.txt --outfile ./dataset/long_summarization_data/recursive-arxiv-dataset/output/val.bin 
  
  python3 -m model.long_summarization.scripts.json_to_bin --infile ./dataset/long_summarization_data/arxiv-dataset/test.txt --outfile ./dataset/long_summarization_data/recursive-arxiv-dataset/output/test.bin 
  ```

##  

- Decode

```
python3 -m model.long_summarization.run_summarization --mode=decode --single_pass=True --data_path=./dataset/long_summarization_data/small-arxiv-dataset/output/train.bin --lr=0.001 --vocab_path=./dataset/long_summarization_data/small-arxiv-dataset/vocab --log_root=./checkpoint/long_summarization/logroot --exp_name=large-pretrained-vocab-experiment --max_dec_steps=100 --max_enc_steps=1600 --num_sections=4 --max_section_len=400 --batch_size=4 --vocab_size=200000 --use_do=True --optimizer=adagrad --do_prob=0.25 --hier=True --split_intro=True --fixed_attn=True --legacy_encoder=False --coverage=False
```



- Link dataset: https://drive.google.com/file/d/1b3rmCSIoh6VhD4HKWjI4HOW-cSwcwbeC/view (create a folder dataset and extract it inside)
- Link checkpoint: https://drive.google.com/file/d/1Rpl4yKhFV-YZFq2uPW_8J2e-nqIwZVKy/view?usp=sharing (create a folder checkpoint and extract it inside)