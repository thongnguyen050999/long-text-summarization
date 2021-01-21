from transformers import (
    AdamW,
    PegasusForConditionalGeneration,
    PegasusTokenizer,
    get_linear_schedule_with_warmup
)
from nlp import load_metric
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import torch
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from nlp import list_datasets, load_dataset
import matplotlib.pyplot as plt
import argparse
import glob
import os
import json
import time
import logging
import random
import re
from itertools import chain
from string import punctuation
import pickle
from datasets import Dataset

import nltk
import wandb
wandb.init(project='pegasus')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


set_seed(42)


class PegasusFineTuner(pl.LightningModule):
    def __init__(self, hparams):
        super(PegasusFineTuner, self).__init__()
        self.hparams = hparams
        self.model = PegasusForConditionalGeneration.from_pretrained(
            self.hparams.model_name_or_path)
        self.model = self.model.to(memory_format=torch.channels_last)
        self.tokenizer = PegasusTokenizer.from_pretrained(
            self.hparams.tokenizer_name_or_path)
        self.rouge_metric = load_metric('rouge')

        if self.hparams.freeze_embeds:
            self.freeze_embeds()

        if self.hparams.freeze_encoder:
            self.freeze_params(self.model.get_encoder())
            # assert_all_frozen(self.model.get_encoder())

        n_observations_per_split = {
            'train': self.hparams.n_train,
            'validation': self.hparams.n_val,
            'test': self.hparams.n_test
        }
        self.n_obs = {k: v if v >= 0 else None for k,
                      v in n_observations_per_split.items()}

    def freeze_params(self, model):
        for par in model.parameters():
            par.requires_grad = False

    def freeze_embeds(self):
        try:
            self.freeze_params(self.model.model.shared)
            for d in [self.model.model.encoder, self.model.model.decoder]:
                self.freeze_params(d.embed_positions)
                self.freeze_params(d.embed_tokens)

        except AttributeError:
            self.freeze_params(self.model.model.shared)
            for d in [self.model.model.encoder, self.model.model.decoder]:
                self.freeze_params(d.embed_tokens)

    def lmap(self, f, x):
        return list(map(f, x))

    def is_logger(self):
        return self.trainer.proc_rank <= 0

    def parse_score(self, result):
        return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )

    def _step(self, batch):
        labels = batch['target_ids']
        # labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch['source_ids'],
            attention_mask=batch['source_mask'],
            labels=labels,
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0]
        wandb.log({"loss": loss})
        return loss

    def ids_to_clean_text(self, generated_ids):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return self.lmap(str.strip, gen_text)

    def _generative_step(self, batch):
        t0 = time.time()

        generated_ids = self.model.generate(
            batch['source_ids'],
            attention_mask=batch['source_mask'],
            use_cache=True,
            decoder_attention_mask=batch['target_mask'],
            max_length=256,
            num_beams=1,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True
        )
        preds = self.ids_to_clean_text(generated_ids)
        target = self.ids_to_clean_text(batch['target_ids'])

        gen_time = (time.time() - t0)/batch['source_ids'].shape[0]
        loss = self._step(batch)
        base_metrics = {'val_loss': loss}
        summ_len = np.mean(self.lmap(len, generated_ids))
        base_metrics.update(gen_time=gen_time,
                            gen_len=summ_len, preds=preds, target=target)
        self.rouge_metric.add_batch(preds, target)

        return base_metrics

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        tensorboard_logs = {'training_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_train_loss': avg_train_loss}
        return {'avg_train_loss': avg_train_loss, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        return self._generative_step(batch)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}

        rouge_results = self.rouge_metric.compute()
        rouge_dict = self.parse_score(rouge_results)

        tensorboard_logs.update(
            rouge1=rouge_dict['rouge1'], rougeL=rouge_dict['rougeL'])

        self.target_gen = []
        self.prediction_gen = []
        return {
            'avg_val_loss': avg_loss,
            'rouge1': rouge_results['rouge1'],
            'rougeL': rouge_results['rougeL'],
            'log': tensorboard_logs,
            'progress_bar': tensorboard_logs
        }

    def configure_optimizers(self):
        model = self.model
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.hparams.weight_decay
            },
            {
                'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None, using_native_amp=False):
        if self.trainer.use_tpu:
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {'loss': '{:.3f}'.format(
            self.trainer.avg_loss), 'lr': self.lr_scheduler.get_last_lr()[-1]}
        return tqdm_dict

    def train_dataloader(self):
        n_samples = self.n_obs['train']
        train_dataset = get_dataset(
            tokenizer=self.tokenizer, type_path='train', num_samples=n_samples, args=self.hparams)
        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size,
                                drop_last=True, shuffle=True, num_workers=4)
        t_total = (
            (len(dataloader.dataset)//(self.hparams.train_batch_size*max(1, self.hparams.n_gpu))
             ) // self.hparams.gradient_accumulation_steps*float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total)
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        n_samples = self.n_obs['validation']
        validation_dataset = get_dataset(
            tokenizer=self.tokenizer, type_path='validation', num_samples=n_samples, args=self.hparams)
        return DataLoader(validation_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4)

    def test_dataloader(self):
        n_samples = self.n_obs['test']
        test_dataset = get_dataset(
            tokenizer=self.tokenizer, type_path='test', num_samples=n_samples, args=self.hparams)
        return DataLoader(test_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4)


logger = logging.getLogger(__name__)


class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info('***** Validation results *****')
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            for key in sorted(metrics):
                if key not in ['log', 'progress_bar']:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info('***** Test results *****')

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

            output_test_results = os.path.join(
                pl_module.hparams.output_dir, 'test_results.txt')
            with open(output_test_resutls, 'w') as writer:
                for key in sorted(metrics):
                    if key not in ['log', 'progress_bar']:
                        logger.info("{} = {}\n".format(key, str(metrics[key])))
                        writer.write("{} = {}\n".format(
                            key, str(metrics[key])))


class ArXiv(torch.utils.data.Dataset):
    def __init__(self, tokenizer, type_path, num_samples, input_length, output_length, print_text=False):
        if type_path == 'train': data_files = 'train.pkl'
        elif type_path == 'validation': data_files = 'val.pkl'
        else: data_files = 'test.pkl'
        with open(os.path.join('./data/arxiv-dataset',data_files),'rb') as f:
            data_dict = pickle.load(f)
            
        self.dataset = Dataset.from_dict(data_dict)
        if num_samples:
            self.dataset = self.dataset.select(list(range(0,num_samples)))
        self.input_length = input_length
        self.tokenizer = tokenizer
        self.output_length = output_length
        self.print_text = print_text

    def __len__(self):
        return self.dataset.shape[0]

    def clean_text(self, text):
        text = text.replace('<S>','')
        text = text.replace('</S>','')
        text = text.strip()
        text = text.replace('\n', '')
        text = text.replace('  ', ' ')
        return text

    def convert_to_features(self, example_batch):
        if self.print_text:
            print('Input Text:', self.clean_text(example_batch['article']))

        input_ = self.clean_text(example_batch['article'])
        target_ = self.clean_text(example_batch['abstract'])

        source = self.tokenizer.batch_encode_plus(
            [input_], max_length=self.input_length, padding='max_length', truncation=True, return_tensors='pt')
        targets = self.tokenizer.batch_encode_plus(
            [target_], max_length=self.output_length, padding='max_length', truncation=True, return_tensors='pt')

        return source, targets

    def __getitem__(self, index):
        source, targets = self.convert_to_features(self.dataset[index])
        source_ids = source['input_ids'].squeeze()
        target_ids = targets['input_ids'].squeeze()

        src_mask = source['attention_mask'].squeeze()
        target_mask = targets['attention_mask'].squeeze()
        
        return {'source_ids': source_ids, 'source_mask': src_mask, 'target_ids': target_ids, 'target_mask': target_mask}


def get_dataset(tokenizer, type_path, num_samples, args):
    return ArXiv(tokenizer=tokenizer, type_path=type_path, num_samples=num_samples, input_length=args.max_input_length, output_length=args.max_output_length)


def main():
    args_dict = dict(
        output_dir="./arxiv",
        model_name_or_path="google/pegasus-large",
        tokenizer_name_or_path="google/pegasus-large",
        max_input_length=1024,
        max_output_length=256,
        freeze_encoder=True,
        freeze_embeds=True,
        learning_rate=1e-4,
        weight_decay=0.0,
        adam_epsilon=1e-8,
        warmup_steps=0,
        num_train_epochs=20,
        gradient_accumulation_steps=8,
        n_gpu=1,
        resume_from_checkpoint=None,
        n_val=-1,
        n_train=-1,
        n_test=-1,
        early_stop_callback=False,
        fp_16=False,
        opt_level='O1',
        max_grad_norm=1.0,
        seed=42,
        check_val_every_n_epoch=2
    )

    args_dict.update({'output_dir': './arxiv', 'num_train_epochs': 20,
                      'train_batch_size': 8, 'eval_batch_size': 8})
    args = argparse.Namespace(**args_dict)
    print(args_dict)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=args.output_dir,
        prefix='checkpoint',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
    )

    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.n_gpu,
        max_epochs=args.num_train_epochs,
        early_stop_callback=False,
        precision=16 if args.fp_16 else 32,
        amp_level=args.opt_level,
        resume_from_checkpoint=args.resume_from_checkpoint,
        gradient_clip_val=args.max_grad_norm,
        checkpoint_callback=checkpoint_callback,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        callbacks=[LoggingCallback()],
    )
    model = PegasusFineTuner(args)
    trainer = pl.Trainer(**train_params)
    trainer.fit(model)


if __name__ == '__main__':
    main()
