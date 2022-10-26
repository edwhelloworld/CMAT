from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F
from my_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from my_bert.mner_modeling import *#(CONFIG_NAME, WEIGHTS_NAME,BertConfig, MTCCMBertForMMTokenClassificationCRF)
from my_bert.optimization import BertAdam, warmup_linear
from my_bert.tokenization import BertTokenizer
from seqeval.metrics import classification_report
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

import resnet.resnet as resnet
from resnet.resnet_utils import myResnet

from torchvision import transforms
from PIL import Image

from sklearn.metrics import precision_recall_fscore_support

from ner_evaluate import evaluate_each_class
from ner_evaluate import evaluate
from process_data import *#_entity import *


def macro_f1(y_true, y_pred):
    p_macro, r_macro, f_macro, support_macro \
        = precision_recall_fscore_support(y_true, y_pred, average='macro')
    f_macro = 2 * p_macro * r_macro / (p_macro + r_macro)
    return p_macro, r_macro, f_macro


def main():
    import myconfig
    args = myconfig #get_args()

    if args.task_name == "twitter2017":
        args.path_image = args.path_image + "/twitter2017_images/"
    elif args.task_name == "twitter2015":
        args.path_image = args.path_image + "/twitter2015_images/"

    if args.server_ip and args.server_port:
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    processors = {
        "twitter2015": MNERProcessor,
        "twitter2017": MNERProcessor
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()
    auxlabel_list = processor.get_auxlabels()
    num_labels = len(label_list) + 1  # label 0 corresponds to padding, label in label_list starts from 1
    auxnum_labels = len(auxlabel_list) + 1  # label 0 corresponds to padding, label in label_list starts from 1

    start_label_id = processor.get_start_label_id()
    stop_label_id = processor.get_stop_label_id()

    # ''' initialization of our conversion matrix, in our implementation, it is a 7*12 matrix initialized as follows:
    trans_matrix = np.zeros((auxnum_labels, num_labels), dtype=float)#wo train? no change after training
    trans_matrix[0, 0] = 1  # pad to pad
    trans_matrix[1, 1] = 1  # O to O
    trans_matrix[2, 2] = 0.25  # B to B-MISC
    trans_matrix[2, 4] = 0.25  # B to B-PER
    trans_matrix[2, 6] = 0.25  # B to B-ORG
    trans_matrix[2, 8] = 0.25  # B to B-LOC
    trans_matrix[3, 3] = 0.25  # I to I-MISC
    trans_matrix[3, 5] = 0.25  # I to I-PER
    trans_matrix[3, 7] = 0.25  # I to I-ORG
    trans_matrix[3, 9] = 0.25  # I to I-LOC
    trans_matrix[4, 10] = 1  # X to X
    trans_matrix[5, 11] = 1  # [CLS] to [CLS]
    trans_matrix[6, 12] = 1  # [SEP] to [SEP]

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                   'distributed_{}'.format(args.local_rank))

    if args.mm_model == 'MTCCMBert':
        model = MTCCMBertForMMTokenClassificationCRF.from_pretrained(args.bert_model,
                                                                     cache_dir=cache_dir, layer_num1=args.layer_num1,
                                                                     layer_num2=args.layer_num2,
                                                                     layer_num3=args.layer_num3,
                                                                     num_labels=num_labels, auxnum_labels=auxnum_labels)
    else:
        print('please define your MNER Model')

    net = getattr(resnet, 'resnet152')()
    net.load_state_dict(torch.load(os.path.join(args.resnet_root, 'resnet152.pth')))
    encoder = myResnet(net, args.fine_tune_cnn, device)
    mi_net = MIDiscriminator(x_channels=768, z_channels=768)

    if args.fp16:
        model.half()
        encoder.half()
        mi_net.half()
    model.to(device)
    encoder.to(device)
    mi_net.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
        encoder = DDP(encoder)
        mi_net = DDP(mi_net)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
        encoder = torch.nn.DataParallel(encoder)
        mi_net = torch.nn.DataParallel(mi_net)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': encoder.parameters(), 'lr': args.cnn_lr},
        # {'params': [p for n, p in list(encoder.named_parameters()) if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01}
        {'params': mi_net.parameters(), 'lr': args.mi_lr}
    ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    output_encoder_file = os.path.join(output_dir, "pytorch_encoder.bin")

    if args.do_train:
        train_features = convert_mm_examples_to_features(train_examples, label_list, auxlabel_list, args.max_seq_length, tokenizer, args.crop_size, args.path_image)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_added_input_mask = torch.tensor([f.added_input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_img_feats = torch.stack([f.img_feat for f in train_features])
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_auxlabel_ids = torch.tensor([f.auxlabel_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_added_input_mask, all_segment_ids, all_img_feats, all_label_ids, all_auxlabel_ids)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        if args.testData_asDev:
            eval_examples = processor.get_test_examples(args.data_dir)
        else:
            eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_mm_examples_to_features(eval_examples, label_list, auxlabel_list, args.max_seq_length, tokenizer, args.crop_size, args.path_image)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_added_input_mask = torch.tensor([f.added_input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_img_feats = torch.stack([f.img_feat for f in eval_features])
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_auxlabel_ids = torch.tensor([f.auxlabel_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_added_input_mask, all_segment_ids, all_img_feats, all_label_ids, all_auxlabel_ids)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # test_eval_examples = processor.get_test_examples(args.data_dir)
        # test_eval_features = convert_mm_examples_to_features(test_eval_examples, label_list, auxlabel_list, args.max_seq_length, tokenizer, args.crop_size, args.path_image)
        # all_input_ids = torch.tensor([f.input_ids for f in test_eval_features], dtype=torch.long)
        # all_input_mask = torch.tensor([f.input_mask for f in test_eval_features], dtype=torch.long)
        # all_added_input_mask = torch.tensor([f.added_input_mask for f in test_eval_features], dtype=torch.long)
        # all_segment_ids = torch.tensor([f.segment_ids for f in test_eval_features], dtype=torch.long)
        # all_img_feats = torch.stack([f.img_feat for f in test_eval_features])
        # all_label_ids = torch.tensor([f.label_id for f in test_eval_features], dtype=torch.long)
        # all_auxlabel_ids = torch.tensor([f.auxlabel_id for f in test_eval_features], dtype=torch.long)
        # test_eval_data = TensorDataset(all_input_ids, all_input_mask, all_added_input_mask, all_segment_ids, all_img_feats, all_label_ids, all_auxlabel_ids)
        #
        # # Run prediction for full data
        # test_eval_sampler = SequentialSampler(test_eval_data)
        # test_eval_dataloader = DataLoader(test_eval_data, sampler=test_eval_sampler, batch_size=args.eval_batch_size)

        max_dev_f1 = 0.0
        max_test_f1 = 0.0
        best_dev_epoch = 0
        best_test_epoch = 0
        logger.info("***** Running training *****")
        for train_idx in trange(int(args.num_train_epochs), desc="Epoch"):
            logger.info("********** Epoch: " + str(train_idx) + " **********")
            logger.info("  Num examples = %d", len(train_examples))
            logger.info("  Batch size = %d", args.train_batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps)
            model.train()
            encoder.train()
            encoder.zero_grad()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, added_input_mask, segment_ids, img_feats, label_ids, auxlabel_ids = batch
                with torch.no_grad():
                    imgs_f, img_mean, img_att = encoder(img_feats)
                trans_matrix = torch.tensor(trans_matrix).to(device)
                neg_log_likelihood = model(input_ids, segment_ids, input_mask, added_input_mask, img_att, trans_matrix, label_ids, auxlabel_ids, mi_net, device)
                #负对数似然
                if n_gpu > 1:
                    neg_log_likelihood = neg_log_likelihood.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    neg_log_likelihood = neg_log_likelihood / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(neg_log_likelihood)
                else:
                    neg_log_likelihood.backward()

                tr_loss += neg_log_likelihood.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses. if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_optimization_steps, args.warmup_proportion)

                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
#todo:============================================ Eval ============================================
            logger.info("***** Running evaluation on Dev Set*****")
            logger.info("  Num examples = %d", len(eval_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)

            model.eval()
            encoder.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            y_true = []
            y_pred = []
            label_map = {i: label for i, label in enumerate(label_list, 1)}
            label_map[0] = "PAD"
            for input_ids, input_mask, added_input_mask, segment_ids, img_feats, label_ids, auxlabel_ids in tqdm(eval_dataloader, desc="Evaluating"):
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                added_input_mask = added_input_mask.to(device)
                segment_ids = segment_ids.to(device)
                img_feats = img_feats.to(device)
                label_ids = label_ids.to(device)
                auxlabel_ids = auxlabel_ids.to(device)
                # trans_matrix = torch.tensor(trans_matrix).to(device)

                with torch.no_grad():
                    imgs_f, img_mean, img_att = encoder(img_feats)
                    predicted_label_seq_ids = model(input_ids, segment_ids, input_mask, added_input_mask, img_att, trans_matrix)

                # logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
                # logits = logits.detach().cpu().numpy()
                # logits = predicted_label_seq_ids.detach().cpu().numpy()
                logits = predicted_label_seq_ids
                label_ids = label_ids.to('cpu').numpy()
                input_mask = input_mask.to('cpu').numpy()
                for i, mask in enumerate(input_mask):
                    temp_1 = []
                    temp_2 = []
                    for j, m in enumerate(mask):
                        if j == 0:
                            continue
                        if m:
                            if label_map[label_ids[i][j]] != "X" and label_map[label_ids[i][j]] != "[SEP]":
                                temp_1.append(label_map[label_ids[i][j]])
                                temp_2.append(label_map[logits[i][j]])
                        else:
                            # temp_1.pop()
                            # temp_2.pop()
                            break
                    y_true.append(temp_1)
                    y_pred.append(temp_2)
            report = classification_report(y_true, y_pred, digits=4)
            logger.info("*****Epoch: " + str(train_idx)+ " Dev Eval results *****")
            logger.info("\n%s", report)
            # eval_true_label = np.concatenate(y_true)
            # eval_pred_label = np.concatenate(y_pred)
            # precision, recall, F_score = macro_f1(eval_true_label, eval_pred_label)
            F_score_dev = float(report.split('\n')[-4].split('      ')[-2].split('    ')[-1])#[-4]micro f1#float(report.split('\n')[-3].split('      ')[-2].split('    ')[-1])#[-3]macro avg F1
            logger.info("report[-4]F-score: {}".format(str(F_score_dev)))#print("F-score: ", F_score_dev)
            #logger.info("TransMatrix: {}".format(str(trans_matrix)))
#todo:============================================= Test ==========================================================
            # logger.info("***** Running Test *****")
            # logger.info("  Num examples = %d", len(test_eval_examples))
            # logger.info("  Batch size = %d", args.eval_batch_size)
            # y_true = []
            # y_pred = []
            # label_map = {i: label for i, label in enumerate(label_list, 1)}
            # label_map[0] = "PAD"
            # for input_ids, input_mask, added_input_mask, segment_ids, img_feats, label_ids, auxlabel_ids in tqdm(test_eval_dataloader, desc="Evaluating"):
            #     input_ids = input_ids.to(device)
            #     input_mask = input_mask.to(device)
            #     added_input_mask = added_input_mask.to(device)
            #     segment_ids = segment_ids.to(device)
            #     img_feats = img_feats.to(device)
            #     label_ids = label_ids.to(device)
            #     auxlabel_ids = auxlabel_ids.to(device)
            #     # trans_matrix = torch.tensor(trans_matrix).to(device)
            #
            #     with torch.no_grad():
            #         imgs_f, img_mean, img_att = encoder(img_feats)
            #         predicted_label_seq_ids = model(input_ids, segment_ids, input_mask, added_input_mask, img_att, trans_matrix)
            #
            #     # logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
            #     # logits = logits.detach().cpu().numpy()
            #     # logits = predicted_label_seq_ids.detach().cpu().numpy()
            #     logits = predicted_label_seq_ids
            #     label_ids = label_ids.to('cpu').numpy()
            #     input_mask = input_mask.to('cpu').numpy()
            #     for i, mask in enumerate(input_mask):
            #         temp_1 = []
            #         temp_2 = []
            #         for j, m in enumerate(mask):
            #             if j == 0:
            #                 continue
            #             if m:
            #                 if label_map[label_ids[i][j]] != "X" and label_map[label_ids[i][j]] != "[SEP]":
            #                     temp_1.append(label_map[label_ids[i][j]])
            #                     temp_2.append(label_map[logits[i][j]])
            #             else:
            #                 # temp_1.pop()
            #                 # temp_2.pop()
            #                 break
            #         y_true.append(temp_1)
            #         y_pred.append(temp_2)
            # report = classification_report(y_true, y_pred, digits=4)
            # logger.info("********** Epoch: " + str(train_idx) +"!!Test!! results *****")
            # logger.info("\n%s", report)
            # F_score_test = float(report.split('\n')[-3].split('      ')[-2].split('    ')[-1])

            if F_score_dev > max_dev_f1:
                # Save a trained model and the associated configuration
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                encoder_to_save = encoder.module if hasattr(encoder, 'module') else encoder  # Only save the model it-self
                torch.save(model_to_save.state_dict(), output_model_file)
                torch.save(encoder_to_save.state_dict(), output_encoder_file)
                with open(output_config_file, 'w') as f:
                    f.write(model_to_save.config.to_json_string())
                label_map = {i: label for i, label in enumerate(label_list, 1)}
                model_config = {"bert_model": args.bert_model, "do_lower": args.do_lower_case,
                                "max_seq_length": args.max_seq_length, "num_labels": len(label_list) + 1, "label_map": label_map}
                json.dump(model_config, open(os.path.join(output_dir, "model_config.json"), "w"))
                max_dev_f1 = F_score_dev
                best_dev_epoch = train_idx
            # if F_score_test > max_test_f1:
            #     max_test_f1 = F_score_test
            #     best_test_epoch = train_idx

        logger.info("**************************************************")
        logger.info("The best epoch on the dev set: {}".format(str(best_dev_epoch)))#, str(best_dev_epoch))
        logger.info("The best Micro-F1 score on the dev set: {}".format(str(max_dev_f1)))#, str(max_dev_f1))
        # logger.info("The best epoch on the test set: {}".format(str(best_test_epoch)))#, str(best_test_epoch))
        # logger.info("The best Micro-F1 score on the test set: {}".format(str(max_test_f1)))#, str(max_test_f1))
        logger.info('\n')

    config = BertConfig(output_config_file)
    if args.mm_model == 'MTCCMBert':
        model = MTCCMBertForMMTokenClassificationCRF(config, layer_num1=args.layer_num1, layer_num2=args.layer_num2,
                                                     layer_num3=args.layer_num3, num_labels=num_labels,
                                                     auxnum_labels=auxnum_labels)
    else:
        print('please define your MNER Model')

    model.load_state_dict(torch.load(output_model_file))
    model.to(device)
    encoder_state_dict = torch.load(output_encoder_file)
    encoder.load_state_dict(encoder_state_dict)
    encoder.to(device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = processor.get_test_examples(args.data_dir)
        eval_features = convert_mm_examples_to_features(eval_examples, label_list, auxlabel_list, args.max_seq_length, tokenizer, args.crop_size, args.path_image)
        logger.info("***** Running ！！Test！！ Evaluation with the Best Model on the Dev Set*****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_added_input_mask = torch.tensor([f.added_input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_img_feats = torch.stack([f.img_feat for f in eval_features])
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_auxlabel_ids = torch.tensor([f.auxlabel_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_added_input_mask, all_segment_ids, all_img_feats, all_label_ids, all_auxlabel_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
        model.eval()
        encoder.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        y_true = []
        y_pred = []
        y_true_idx = []
        y_pred_idx = []
        label_map = {i: label for i, label in enumerate(label_list, 1)}
        label_map[0] = "PAD"
        for input_ids, input_mask, added_input_mask, segment_ids, img_feats, label_ids, auxlabel_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            added_input_mask = added_input_mask.to(device)
            segment_ids = segment_ids.to(device)
            img_feats = img_feats.to(device)
            label_ids = label_ids.to(device)
            auxlabel_ids = auxlabel_ids.to(device)
            trans_matrix = torch.tensor(trans_matrix).to(device)

            with torch.no_grad():
                imgs_f, img_mean, img_att = encoder(img_feats)
                predicted_label_seq_ids = model(input_ids, segment_ids, input_mask, added_input_mask, img_att, trans_matrix)

            # logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
            # logits = logits.detach().cpu().numpy()
            # logits = predicted_label_seq_ids.detach().cpu().numpy()
            logits = predicted_label_seq_ids
            label_ids = label_ids.to('cpu').numpy()
            input_mask = input_mask.to('cpu').numpy()
            for i, mask in enumerate(input_mask):
                temp_1 = []
                temp_2 = []
                tmp1_idx = []
                tmp2_idx = []
                # count = 0
                for j, m in enumerate(mask):
                    if j == 0:
                        continue
                    if m:
                        if label_map[label_ids[i][j]] != "X" and label_map[label_ids[i][j]] != "[SEP]":
                            temp_1.append(label_map[label_ids[i][j]])
                            tmp1_idx.append(label_ids[i][j])
                            temp_2.append(label_map[logits[i][j]])
                            tmp2_idx.append(logits[i][j])
                        # if label_map[label_ids[i][j]].startswith("B"):
                        # count += 1
                    else:
                        # temp_1.pop()
                        # temp_2.pop()
                        break
                y_true.append(temp_1)
                y_pred.append(temp_2)
                y_true_idx.append(tmp1_idx)
                y_pred_idx.append(tmp2_idx)
                # if count > 1:
                # multi_ent_y_true.append(temp_1)
                # multi_ent_y_pred.append(temp_2)

        report = classification_report(y_true, y_pred, digits=4)
        # multi_ent_report = classification_report(multi_ent_y_true, multi_ent_y_pred,digits=4)

        sentence_list = []
        test_data, imgs, _ = processor._read_mmtsv(os.path.join(args.data_dir, "test.txt"))
        output_pred_file = os.path.join(output_dir, "mtmner_pred.txt")
        fout = open(output_pred_file, 'w')
        for i in range(len(y_pred)):
            sentence = test_data[i][0]
            sentence_list.append(sentence)
            img = imgs[i]
            samp_pred_label = y_pred[i]
            samp_true_label = y_true[i]
            fout.write(img + '\n')
            fout.write(' '.join(sentence) + '\n')
            fout.write(' '.join(samp_pred_label) + '\n')
            fout.write(' '.join(samp_true_label) + '\n' + '\n')
        fout.close()

        reverse_label_map = {label: i for i, label in enumerate(label_list, 1)}
        acc, f1, p, r = evaluate(y_pred_idx, y_true_idx, sentence_list, reverse_label_map)
        print("Overall: ", p, r, f1)
        per_f1, per_p, per_r = evaluate_each_class(y_pred_idx, y_true_idx, sentence_list, reverse_label_map, 'PER')
        print("Person: ", per_p, per_r, per_f1)
        loc_f1, loc_p, loc_r = evaluate_each_class(y_pred_idx, y_true_idx, sentence_list, reverse_label_map, 'LOC')
        print("Location: ", loc_p, loc_r, loc_f1)
        org_f1, org_p, org_r = evaluate_each_class(y_pred_idx, y_true_idx, sentence_list, reverse_label_map, 'ORG')
        print("Organization: ", org_p, org_r, org_f1)
        misc_f1, misc_p, misc_r = evaluate_each_class(y_pred_idx, y_true_idx, sentence_list, reverse_label_map, 'MISC')
        print("Miscellaneous: ", misc_p, misc_r, misc_f1)

        output_eval_file = os.path.join(output_dir, "test_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Test results *****")
            logger.info("\n%s", report)
            logger.info("Overall: " + str(p) + ' ' + str(r) + ' ' + str(f1) + '\n')
            writer.write(report)
            writer.write("Overall: " + str(p) + ' ' + str(r) + ' ' + str(f1) + '\n')
            writer.write("Person: " + str(per_p) + ' ' + str(per_r) + ' ' + str(per_f1) + '\n')
            writer.write("Location: " + str(loc_p) + ' ' + str(loc_r) + ' ' + str(loc_f1) + '\n')
            writer.write("Organization: " + str(org_p) + ' ' + str(org_r) + ' ' + str(org_f1) + '\n')
            writer.write("Miscellaneous: " + str(misc_p) + ' ' + str(misc_r) + ' ' + str(misc_f1) + '\n')
            #writer.write("best_test_epoch: " + str(best_test_epoch) + '\n')
            writer.write("best_dev_epoch: " + str(best_dev_epoch) + '\n')

if __name__ == "__main__":
    main()
