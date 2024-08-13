import random
import pandas as pd
import torch
from torch.utils.data import Dataset

path = '/store01/nchawla/kle3/MolX-24/datasets/pretraining/molecule/translation/'

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

class BaselineDatasetGeneration(Dataset):
    def __init__(self, data, split, task="", augment=False, icl=0, use_smiles=False, use_llasmol=False, use_t5chem=False):
        super().__init__()
        self.split = split
        self.task = task
        if self.task == "Reactants":
            path = '/store01/nchawla/kle3/MolX-24/datasets/downstream/molecule/retrosynthesis/'
        elif self.task == "Modified SMILES":
            path = '/store01/nchawla/kle3/MolX-24/datasets/downstream/molecule/optimization'
        else:
            path = '/store01/nchawla/kle3/MolX-24/datasets/pretraining/molecule/translation/'
        self.questions = pd.read_csv(f'{path}/{data}/{split}.csv')
        if self.split == "pretrain" and use_smiles:
            self.questions = self.questions[self.questions["SMILES"].str.len() <= 200].reset_index(drop=True)
        if data == "PubChem324k": 
            if task == "Description":
                self.Text = "provide a detailed description of that molecule"
                self.Pred = "Description"
            if task == "Canonical SMILES":
                self.Text = "provide the molecule’s canonical SMILES string, which is a unique representation of that molecule"
                self.Pred = "Canonical SMILES"
                self.questions = self.questions.sample(frac=0.1, random_state=42).reset_index(drop=True)
            if task == "IUPAC name":
                self.Text = "provide the molecule’s IUPAC name"
                self.Pred = "IUPAC name"
                self.questions = self.questions[self.questions[self.Pred] != "unknown"].reset_index(drop=True)
        if task == "Reactants":
            if use_t5chem:
                self.Text = "Predict the reaction that produces the following product"
            else:
                self.Text = "provide SMILES strings of possible reactants used in the molecule’s synthesis. The reactants should be split by '.'"
            self.Pred = "Reactants"

        self.augment = augment
        self.icl = icl
        self.use_smiles = use_smiles
        self.use_llasmol = use_llasmol
        self.use_t5chem = use_t5chem
        if self.icl > 0:
            self.train = pd.read_csv(f'{path}/{data}/train.csv')
            self.train = self.train[self.train[self.Pred] != "unknown"].reset_index(drop=True)
            self.train = self.train[self.train[self.Pred].str.len() <= 200].reset_index(drop=True)
            self.train["Fingerprint"] = self.train["SMILES"].apply(get_fp)
            self.train = self.train.dropna(subset=['Fingerprint'])

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.questions)

    def __getitem__(self, index):
        data = self.questions.iloc[index]
        if self.task != "Modified SMILES":
            Text = self.Text
            Pred = self.Pred
        else:
            Text = data["Text"]
            Pred = "Modified SMILES"
        item_SMILES = data["SMILES"]
        if self.augment:
            if random.random() <= 0.5:
                try:
                    item_mol = Chem.MolFromSmiles(item_SMILES)
                    item_SMILES = Chem.MolToSmiles(item_mol, doRandom=True)
                except:
                    pass

        if self.split != "pretrain":
            icl = self.icl
        else:
            icl = random.randint(0, self.icl)
        if icl > 0:
            self.examples = search_examples(item_SMILES, self.train, icl)
            question = "Given the SMILES string of a molecule, {}. You are provided with {} examples, please strictly follow the format of examples, no need to repeat the examples and no other information can be provided. ".format(Text, icl)
            # question += "\nExamples: "
            for i in range(len(self.examples)):
                question += "\nSMILES: {}".format(self.examples.loc[i, "SMILES"])
                question += "\n{}: {}".format(Pred, str(self.examples.loc[i, Pred]))
            # question += "\n\nQuestion: "
            question += "\nSMILES: {}".format(item_SMILES)
            question += "\n{}: ".format(Pred)
        else:
            if self.use_t5chem:
                question = "{}".format(item_SMILES)
            else:
                question = "Given the SMILES string of a molecule, {}. ".format(Text)
                if not self.use_llasmol:
                    question += "SMILES: {}".format(item_SMILES)
                    question += "\n\n{}: ".format(Pred)
                else:
                    question += "<SMILES> {} </SMILES>".format(item_SMILES)

        return {
            'id': index,
            'question': question,
            'label': str(data[Pred]),
        }