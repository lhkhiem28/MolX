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

class BaselineDatasetRegression(Dataset):
    def __init__(self, data, split, task="Label", icl=0, use_smiles=False, use_llasmol=False, use_t5chem=False):
        super().__init__()
        self.split = split
        self.task = task
        if self.split == "pretrain":
            path = '/store01/nchawla/kle3/MolX-24/datasets/pretraining/molecule/translation/'
        else:
            path = '/store01/nchawla/kle3/MolX-24/datasets/downstream/molecule/prediction/property/'
        self.questions = pd.read_csv(f'{path}/{data}/{split}.csv')

        if task == "Heavy Atom": 
            self.Text = "A heavy atom refers to any atom that is not hydrogen. How many heavy atoms are there in that molecule?"
            self.Pred = "Answer"
            self.questions = self.questions.sample(frac=0.1, random_state=42).reset_index(drop=True)
        if task == "Hydrogen Bond Acceptor": 
            self.Text = "A hydrogen bond acceptor has lone electron pairs that help form hydrogen bonds. How many hydrogen bond acceptors are there in that molecule?"
            self.Pred = "Answer"
            self.questions = self.questions.sample(frac=0.1, random_state=42).reset_index(drop=True)
        if task == "Hydrogen Bond Donor": 
            self.Text = "A hydrogen bond donor is a compound that donates protons (hydrogen atoms) covalently bound to itself, allowing it to form hydrogen bonds. How many hydrogen bond donors are there in that molecule?"
            self.Pred = "Answer"
            self.questions = self.questions.sample(frac=0.1, random_state=42).reset_index(drop=True)
        if task == "Rotatable Bond": 
            self.Text = "A rotatable bond is any single non-ring bond, attached to a non-terminal, non-hydrogen atom. How many rotatable bonds are there in that molecule?"
            self.Pred = "Answer"
            self.questions = self.questions.sample(frac=0.1, random_state=42).reset_index(drop=True)
        if task == "Aromatic Ring": 
            self.Text = "Aromatic rings are hydrocarbons with a benzene or related ring. How many aromatic rings are there in that molecule?"
            self.Pred = "Answer"
            self.questions = self.questions.sample(frac=0.1, random_state=42).reset_index(drop=True)
        if task == "Topological Polar Surface Area": 
            self.Text = "The topological polar surface area (TPSA) is the surface sum of all polar atoms or molecules, primarily oxygen and nitrogen, also including their attached hydrogen atoms. What is the TPSA value of that molecule?"
            self.Pred = "Answer"
            self.questions = self.questions.sample(frac=0.1, random_state=42).reset_index(drop=True)
        if task == "Weight": 
            self.Text = "The molecular weight is the sum of the atomic weights of all the atoms in the molecule. What is the molecular weight of that molecule?"
            self.Pred = "Answer"
            self.questions = self.questions.sample(frac=0.1, random_state=42).reset_index(drop=True)
        if task == "LogP": 
            self.Text = "LogP, or octanol-water partition coefficient, is a measure of how hydrophilic or hydrophobic a molecule is. What is the LogP value of that molecule?"
            self.Pred = "Answer"
            self.questions = self.questions.sample(frac=0.1, random_state=42).reset_index(drop=True)
        if task == "Quantitative Estimate of Druglikeness": 
            self.Text = "The quantitative estimate of druglikeness (QED) is a measure of how drug-like a molecule is, based on various molecular properties associated with druglikeness. What is the QED value of that molecule?"
            self.Pred = "Answer"
            self.questions = self.questions.sample(frac=0.1, random_state=42).reset_index(drop=True)

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
            question = "Given the SMILES string of a molecule, answer the following question. {} You are provided with {} examples, please answer the question with a numerical value only, if uncertain, make an estimate, no need to repeat the examples and no other information can be provided. ".format(Text, self.icl)
            # question += "\nExamples: "
            for i in range(len(self.examples)):
                question += "\nSMILES: {}".format(self.examples.loc[i, "SMILES"])
                question += "\n{}: {}".format(Pred, str(self.examples.loc[i, self.task]))
            # question += "\n\nQuestion: "
            question += "\nSMILES: {}".format(data["SMILES"])
            question += "\n{}: ".format(Pred)
        else:
            if self.use_t5chem:
                question = "{}".format(data["SMILES"])
            else:
                question = "Given the SMILES string of a molecule, answer the following question. {} Please answer the question with a numerical value only, if uncertain, make an estimate, no other information can be provided. ".format(Text)
                if not self.use_llasmol:
                    question += "SMILES: {}".format(data["SMILES"])
                    question += "\n\n{}: ".format(Pred)
                else:
                    question += "<SMILES> {} </SMILES>".format(data["SMILES"])

        return {
            'id': index,
            'question': question,
            'label': str(data[self.task]),
        }