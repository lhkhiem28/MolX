import json
import pandas as pd
from sklearn import metrics
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from Levenshtein import distance
import numpy as np
import re
import tqdm

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit.Chem import MACCSkeys

def get_accuracy_prediction(eval_output, path):
    # eval_output is a list of dicts
    df = pd.concat([pd.DataFrame(d) for d in eval_output])

    # save to csv
    with open(path, 'w')as f:
        for _, row in df.iterrows():
            f.write(json.dumps(dict(row))+'\n')

    predictions, targets = df['pred'].values.tolist(), df['label'].values.tolist()
    pattern = r'\b(Yes|No)\b'
    re_predictions, re_targets = [], []
    for pred, target in zip(predictions, targets):
        match = re.search(pattern, pred)
        if match:
            answer = match.group(0)
            re_predictions.append(str(answer))
            re_targets.append(str(target))

    re_predictions = [1 if "Yes" in x else 0 for x in re_predictions]
    re_targets = [1 if "Yes" in x else 0 for x in re_targets]

    # compute accuracy
    acc = metrics.accuracy_score(
        re_targets, re_predictions, 
    )
    f1 = metrics.f1_score(
        re_targets, re_predictions, 
        average = "macro", 
    )

    return acc, f1, 100*(1 - len(re_predictions)/len(predictions))

def get_rmse_regression(eval_output, path):
    # eval_output is a list of dicts
    df = pd.concat([pd.DataFrame(d) for d in eval_output])

    # save to csv
    with open(path, 'w')as f:
        for _, row in df.iterrows():
            f.write(json.dumps(dict(row))+'\n')

    predictions, targets = df['pred'].values.tolist(), df['label'].values.tolist()
    pattern = r"-?\d+\.?\d*e??\d*?"
    re_predictions, re_targets = [], []
    for pred, target in zip(predictions, targets):
        match = re.search(pattern, pred)
        if match:
            number = match.group(0)
            re_predictions.append(float(number))
            re_targets.append(float(target))

    rmse = metrics.root_mean_squared_error(re_targets, re_predictions)

    return rmse, 100*(1 - len(re_predictions)/len(predictions))

def get_scores_generation(eval_output, path, tokenizer, text_trunc_length, task):
    # eval_output is a list of dicts
    df = pd.concat([pd.DataFrame(d) for d in eval_output])

    # save to csv
    with open(path, 'w')as f:
        for _, row in df.iterrows():
            f.write(json.dumps(dict(row))+'\n')

    predictions, targets = df['pred'].values.tolist(), df['label'].values.tolist()
    if task in ["Description", "IUPAC name"]:
        references = []
        hypotheses = []
        meteor_scores = []
        for gt, out in zip(targets, predictions):
            gt_tokens = tokenizer.tokenize(gt)
            gt_tokens = list(filter(('[PAD]').__ne__, gt_tokens))
            gt_tokens = list(filter(('[CLS]').__ne__, gt_tokens))
            gt_tokens = list(filter(('[SEP]').__ne__, gt_tokens))

            out_tokens = tokenizer.tokenize(out)
            out_tokens = list(filter(('[PAD]').__ne__, out_tokens))
            out_tokens = list(filter(('[CLS]').__ne__, out_tokens))
            out_tokens = list(filter(('[SEP]').__ne__, out_tokens))

            references.append([gt_tokens])
            hypotheses.append(out_tokens)

            mscore = meteor_score([gt_tokens], out_tokens)
            meteor_scores.append(mscore)

        bleu2 = corpus_bleu(references, hypotheses, weights=(.5,.5))
        bleu4 = corpus_bleu(references, hypotheses, weights=(.25,.25,.25,.25))

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
        rouge_scores = []
        for gt, out in tqdm.tqdm(zip(targets, predictions)):
            rs = scorer.score(out, gt)
            rouge_scores.append(rs)

        rouge_1 = np.mean([rs['rouge1'].fmeasure for rs in rouge_scores])
        rouge_2 = np.mean([rs['rouge2'].fmeasure for rs in rouge_scores])
        rouge_l = np.mean([rs['rougeL'].fmeasure for rs in rouge_scores])

        _meteor_score = np.mean(meteor_scores)

        return bleu2, bleu4, rouge_1, rouge_2, rouge_l, _meteor_score
    else:
        matches = []
        references = []
        hypotheses = []
        meteor_scores = []
        distances = []
        for gt, out in zip(targets, predictions):
            if gt == out:
                matches.append(1)
            else:
                matches.append(0)
            gt_tokens = tokenizer.tokenize(gt)
            gt_tokens = list(filter(('[PAD]').__ne__, gt_tokens))
            gt_tokens = list(filter(('[CLS]').__ne__, gt_tokens))
            gt_tokens = list(filter(('[SEP]').__ne__, gt_tokens))

            out_tokens = tokenizer.tokenize(out)
            out_tokens = list(filter(('[PAD]').__ne__, out_tokens))
            out_tokens = list(filter(('[CLS]').__ne__, out_tokens))
            out_tokens = list(filter(('[SEP]').__ne__, out_tokens))

            references.append([gt_tokens])
            hypotheses.append(out_tokens)

            mscore = meteor_score([gt_tokens], out_tokens)
            meteor_scores.append(mscore)
            distances.append(distance(gt, out))

        bleu2 = corpus_bleu(references, hypotheses, weights=(.5,.5))
        bleu4 = corpus_bleu(references, hypotheses, weights=(.25,.25,.25,.25))

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
        rouge_scores = []
        for gt, out in tqdm.tqdm(zip(targets, predictions)):
            rs = scorer.score(out, gt)
            rouge_scores.append(rs)

        rouge_1 = np.mean([rs['rouge1'].fmeasure for rs in rouge_scores])
        rouge_2 = np.mean([rs['rouge2'].fmeasure for rs in rouge_scores])
        rouge_l = np.mean([rs['rougeL'].fmeasure for rs in rouge_scores])

        _meteor_score = np.mean(meteor_scores)
        _distance = np.mean(distances)

        validities = []
        MACCS_sims = []
        Morgan_sims = []
        for gt, out in tqdm.tqdm(zip(targets, predictions)):
            mol_gt = Chem.MolFromSmiles(gt)
            mol_out = Chem.MolFromSmiles(out)
            if (mol_gt is not None) and (mol_out is not None):
                validities.append(1)
                MACCS_sims.append(DataStructs.FingerprintSimilarity(MACCSkeys.GenMACCSKeys(mol_gt), MACCSkeys.GenMACCSKeys(mol_out), metric=DataStructs.TanimotoSimilarity))
                Morgan_sims.append(DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprint(mol_gt, 2), AllChem.GetMorganFingerprint(mol_out, 2)))
            else:
                validities.append(0)

        MACCS_sims_score = np.mean(MACCS_sims)
        Morgan_sims_score = np.mean(Morgan_sims)

        return sum(matches)/len(matches), bleu2, _meteor_score, _distance, MACCS_sims_score, Morgan_sims_score, sum(validities)/len(validities)

eval_funcs = {
    'baseline_prediction': get_accuracy_prediction,
    'baseline_regression': get_rmse_regression,
    'baseline_generation': get_scores_generation,
    'molx_prediction': get_accuracy_prediction,
    'molx_regression': get_rmse_regression,
    'molx_generation': get_scores_generation,
}