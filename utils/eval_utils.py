import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_SB, CLAM_MB
from models.univit_mil import univit_mil_from_embeddings
import pdb
import os
import pandas as pd
from utils.utils import *
from utils.core_utils import Accuracy_Logger
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

def initiate_model(args, ckpt_path, device='cuda'):
    print('Init Model')    
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes, "embed_dim": args.embed_dim}
    
    if args.model_size is not None and args.model_type in ['clam_sb', 'clam_mb']:
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type =='clam_sb':
        model = CLAM_SB(**model_dict)
    elif args.model_type =='clam_mb':
        model = CLAM_MB(**model_dict)
    elif args.model_type == 'univit_mil':
        # Uni-ViT-MIL model initialization
        model = univit_mil_from_embeddings(
            embed_dim=args.embed_dim,
            n_classes=args.n_classes,
            n_biomarkers=args.n_biomarkers,
            mil_num_heads=args.mil_num_heads,
            mil_dropout=args.drop_out,
            head_dropout=args.drop_out,
            biomarker_multi_label=getattr(args, 'biomarker_multi_label', False)
        )
    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)

    print_network(model)

    ckpt = torch.load(ckpt_path)
    ckpt_clean = {}
    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
    model.load_state_dict(ckpt_clean, strict=True)

    _ = model.to(device)
    _ = model.eval()
    return model

def eval(dataset, args, ckpt_path):
    model = initiate_model(args, ckpt_path)
    
    print('Init Loaders')
    loader = get_simple_loader(dataset)
    patient_results, test_error, auc, df, _ = summary(model, loader, args)
    print('test_error: ', test_error)
    print('auc: ', auc)
    return model, patient_results, test_error, auc, df

def summary(model, loader, args):
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, results_dict = model(data)
        
        acc_logger.log(Y_hat, label)
        
        probs = Y_prob.cpu().numpy()

        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        
        error = calculate_error(Y_hat, label)
        test_error += error

    del data
    test_error /= len(loader)

    aucs = []
    if len(np.unique(all_labels)) == 1:
        auc_score = -1

    else: 
        if args.n_classes == 2:
            auc_score = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
            for class_idx in range(args.n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))
            if args.micro_average:
                binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
                fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_probs.ravel())
                auc_score = auc(fpr, tpr)
            else:
                auc_score = np.nanmean(np.array(aucs))

    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(args.n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:,c]})
    df = pd.DataFrame(results_dict)
    return patient_results, test_error, auc_score, df, acc_logger


# =============================================================================
# Uni-ViT-MIL Evaluation Functions
# =============================================================================

def eval_univit(dataset, args, ckpt_path):
    """
    Evaluation function for Uni-ViT-MIL multi-task model.
    
    Computes and logs:
    - Classification accuracy and AUROC
    - Biomarker AUROC (binary or multi-class)
    
    Args:
        dataset: Dataset split to evaluate
        args: Argument namespace with model configuration
        ckpt_path: Path to model checkpoint
    
    Returns:
        model: Loaded model
        patient_results: Per-patient results dictionary
        test_error: Classification error rate
        cls_auc: Classification AUROC
        biomarker_auc: Biomarker AUROC
        df: Results dataframe
    """
    model = initiate_model(args, ckpt_path)
    
    print('Init Loaders')
    loader = get_simple_loader(dataset)
    patient_results, test_error, cls_auc, biomarker_auc, df, _, _ = summary_univit(model, loader, args)
    
    print('='*60)
    print('Uni-ViT-MIL Evaluation Results')
    print('='*60)
    print(f'Classification Error: {test_error:.4f}')
    print(f'Classification Accuracy: {1-test_error:.4f}')
    print(f'Classification AUROC: {cls_auc:.4f}')
    print(f'Biomarker AUROC: {biomarker_auc:.4f}')
    print('='*60)
    
    return model, patient_results, test_error, cls_auc, biomarker_auc, df


def summary_univit(model, loader, args):
    """
    Compute comprehensive evaluation metrics for Uni-ViT-MIL.
    
    Evaluates both classification and biomarker prediction tasks.
    
    Args:
        model: Uni-ViT-MIL model
        loader: Data loader
        args: Argument namespace
    
    Returns:
        patient_results: Per-patient results
        test_error: Classification error
        cls_auc: Classification AUROC
        biomarker_auc: Biomarker AUROC
        df: Results dataframe
        cls_acc_logger: Classification accuracy logger
        bio_acc_logger: Biomarker accuracy logger
    """
    cls_acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    bio_acc_logger = Accuracy_Logger(n_classes=args.n_biomarkers)
    model.eval()
    test_error = 0.

    # Arrays for classification metrics
    all_cls_probs = np.zeros((len(loader), args.n_classes))
    all_cls_labels = np.zeros(len(loader))
    all_cls_preds = np.zeros(len(loader))
    
    # Arrays for biomarker metrics
    all_bio_probs = np.zeros((len(loader), args.n_biomarkers))
    all_bio_labels = np.zeros(len(loader))
    all_bio_preds = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    
    for batch_idx, batch_data in enumerate(loader):
        # Handle both multi-task dataset (3 outputs) and standard dataset (2 outputs)
        if len(batch_data) == 3:
            data, label, biomarker_label = batch_data
            biomarker_label = biomarker_label.to(device)
        else:
            data, label = batch_data
            # Fallback: use class label as biomarker label (clamped to valid range)
            biomarker_label = torch.clamp(label, 0, args.n_biomarkers - 1).to(device)
        
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        
        with torch.no_grad():
            outputs = model(data)
            classification_logits = outputs['classification_logits']
            biomarker_logits = outputs['biomarker_logits']
            attention_weights = outputs['attention_weights']
            
            # Classification predictions
            Y_prob = torch.softmax(classification_logits, dim=1)
            Y_hat = torch.argmax(classification_logits, dim=1)
            
            # Biomarker predictions
            Bio_prob = torch.softmax(biomarker_logits, dim=1)
            Bio_hat = torch.argmax(biomarker_logits, dim=1)
        
        # Log accuracies
        cls_acc_logger.log(Y_hat, label)
        bio_acc_logger.log(Bio_hat, biomarker_label)
        
        # Store predictions
        cls_probs = Y_prob.cpu().numpy()
        bio_probs = Bio_prob.cpu().numpy()

        all_cls_probs[batch_idx] = cls_probs
        all_cls_labels[batch_idx] = label.item()
        all_cls_preds[batch_idx] = Y_hat.item()
        
        all_bio_probs[batch_idx] = bio_probs
        all_bio_labels[batch_idx] = biomarker_label.item()
        all_bio_preds[batch_idx] = Bio_hat.item()
        
        # Store patient results
        patient_results.update({
            slide_id: {
                'slide_id': np.array(slide_id), 
                'cls_prob': cls_probs, 
                'cls_label': label.item(),
                'cls_pred': Y_hat.item(),
                'bio_prob': bio_probs,
                'bio_label': biomarker_label.item(),
                'bio_pred': Bio_hat.item()
            }
        })
        
        # Calculate classification error
        error = calculate_error(Y_hat, label)
        test_error += error

    del data
    test_error /= len(loader)

    # Compute Classification AUROC
    if len(np.unique(all_cls_labels)) == 1:
        cls_auc = -1
    else: 
        if args.n_classes == 2:
            cls_auc = roc_auc_score(all_cls_labels, all_cls_probs[:, 1])
        else:
            try:
                binary_labels = label_binarize(all_cls_labels, classes=[i for i in range(args.n_classes)])
                cls_aucs = []
                for class_idx in range(args.n_classes):
                    if class_idx in all_cls_labels:
                        fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_cls_probs[:, class_idx])
                        cls_aucs.append(auc(fpr, tpr))
                    else:
                        cls_aucs.append(float('nan'))
                
                if getattr(args, 'micro_average', False):
                    fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_cls_probs.ravel())
                    cls_auc = auc(fpr, tpr)
                else:
                    cls_auc = np.nanmean(np.array(cls_aucs))
            except Exception as e:
                print(f'Warning: Could not compute classification AUC: {e}')
                cls_auc = -1

    # Compute Biomarker AUROC
    if len(np.unique(all_bio_labels)) == 1:
        biomarker_auc = -1
    else:
        try:
            if args.n_biomarkers == 2:
                # Binary biomarker AUROC
                biomarker_auc = roc_auc_score(all_bio_labels, all_bio_probs[:, 1])
            else:
                # Multi-class biomarker AUROC
                binary_bio_labels = label_binarize(all_bio_labels, classes=[i for i in range(args.n_biomarkers)])
                bio_aucs = []
                for bio_idx in range(args.n_biomarkers):
                    if bio_idx in all_bio_labels:
                        fpr, tpr, _ = roc_curve(binary_bio_labels[:, bio_idx], all_bio_probs[:, bio_idx])
                        bio_aucs.append(auc(fpr, tpr))
                    else:
                        bio_aucs.append(float('nan'))
                biomarker_auc = np.nanmean(np.array(bio_aucs))
        except Exception as e:
            print(f'Warning: Could not compute biomarker AUC: {e}')
            biomarker_auc = -1

    # Build results dataframe
    results_dict = {
        'slide_id': slide_ids, 
        'cls_label': all_cls_labels, 
        'cls_pred': all_cls_preds,
        'bio_label': all_bio_labels,
        'bio_pred': all_bio_preds
    }
    
    # Add classification probabilities
    for c in range(args.n_classes):
        results_dict.update({'cls_p_{}'.format(c): all_cls_probs[:, c]})
    
    # Add biomarker probabilities
    for b in range(args.n_biomarkers):
        results_dict.update({'bio_p_{}'.format(b): all_bio_probs[:, b]})
    
    df = pd.DataFrame(results_dict)
    
    # Print per-class accuracies
    print('\nClassification Accuracy by Class:')
    for i in range(args.n_classes):
        acc, correct, count = cls_acc_logger.get_summary(i)
        print(f'  Class {i}: acc={acc}, correct={correct}/{count}')
    
    print('\nBiomarker Accuracy by Class:')
    for i in range(args.n_biomarkers):
        acc, correct, count = bio_acc_logger.get_summary(i)
        print(f'  Biomarker {i}: acc={acc}, correct={correct}/{count}')
    
    return patient_results, test_error, cls_auc, biomarker_auc, df, cls_acc_logger, bio_acc_logger
