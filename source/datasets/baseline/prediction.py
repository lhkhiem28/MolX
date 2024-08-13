import random
import pandas as pd
import torch
from torch.utils.data import Dataset

path = '/store01/nchawla/kle3/MolX-24/datasets/downstream/molecule/prediction/property/'

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import DataStructs
from rdkit.Chem import rdMolDescriptors

def get_fp(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        fp = rdMolDescriptors.GetMorganFingerprint(scaffold, 2)
        return fp
    except:
        return None

def get_sim(target_fp, fp):
    try:
        tanimoto_similarity = DataStructs.TanimotoSimilarity(target_fp, fp)
        return tanimoto_similarity
    except:
        return None

def search_examples(target_smiles, train, icl=0):
    target_fp = get_fp(target_smiles)
    train["sim"] = train["Fingerprint"].apply(lambda x: get_sim(target_fp, x))
    train = train.dropna(subset=['sim'])

    train = train.sort_values(by='sim', ascending=False)
    examples = train.head(icl)
    examples = examples.reset_index(drop=True)

    return examples

class BaselineDatasetPrediction(Dataset):
    def __init__(self, data, split, task="Label", icl=0, use_smiles=False, use_llasmol=False, use_t5chem=False):
        super().__init__()
        self.split = split
        self.task = task
        self.questions = pd.read_csv(f'{path}/{data}/{split}.csv')

        self.icl = icl
        self.use_smiles = use_smiles
        self.use_llasmol = use_llasmol
        self.use_t5chem = use_t5chem
        if self.icl > 0:
            self.train = pd.read_csv(f'{path}/{data}/train.csv')
            self.train["Fingerprint"] = self.train["SMILES"].apply(get_fp)
            self.train = self.train.dropna(subset=['Fingerprint'])

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.questions)

    def __getitem__(self, index):
        data = self.questions.iloc[index]
        if self.task != "Label":
            Text = self.Text
            Pred = self.Pred
        else:
            Text = data["Text"]
            Pred = "Answer"
        if self.icl > 0:
            self.examples = search_examples(data["SMILES"], self.train, self.icl)
            question = "Given the SMILES string of a molecule, answer the following question. {} You are provided with {} examples, please answer the question with only Yes or No, if uncertain, make a guess, no need to repeat the examples and no other information can be provided. ".format(Text, self.icl)
            # question += "\nExamples: "
            for i in range(len(self.examples)):
                question += "\nSMILES: {}".format(self.examples.loc[i, "SMILES"])
                question += "\n{}: {}".format(Pred, self.examples.loc[i, "Label"])
            # question += "\n\nQuestion: "
            question += "\nSMILES: {}".format(data["SMILES"])
            question += "\n{}: ".format(Pred)
        else:
            if self.use_t5chem:
                question = "{}".format(data["SMILES"])
            else:
                question = "Given the SMILES string of a molecule, answer the following question. {} Please answer the question with only Yes or No, if uncertain, make a guess, no other information can be provided. ".format(Text)
                if not self.use_llasmol:
                    question += "SMILES: {}".format(data["SMILES"])
                    question += "\n\n{}: ".format(Pred)
                else:
                    question += "<SMILES> {} </SMILES>".format(data["SMILES"])

        return {
            'id': index,
            'question': question,
            'label': data['Label'],
        }