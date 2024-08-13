import random
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector

def smiles2data(smiles):
    """ used in MoleculeNetGraphDataset() class
    Converts rdkit mol objects to graph data object in pytorch geometric
    NB: Uses simplified atom and bond features, and represent as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr """
    mol = Chem.MolFromSmiles(smiles)

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = atom_to_feature_vector(atom)
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    if len(mol.GetBonds()) <= 0:  # mol has no bonds
        num_bond_features = 3  # bond type & direction
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
    else:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = bond_to_feature_vector(bond)

            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data

path = '/store01/nchawla/kle3/MolX-24/datasets/downstream/molecule/prediction/property/'

from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import DataStructs
from rdkit.Chem import rdMolDescriptors

def get_fp_vec(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
        fp_vec = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, fp_vec)
        return fp_vec.tolist()
    except:
        return None

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

class MolXDatasetRegression(Dataset):
    def __init__(self, data, split, task="Label", icl=0, use_smiles=False, use_t5chem=False):
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
            question_1 = "Given the molecule, answer the following question. {} You are provided with {} examples, please answer the question with a numerical value only, if uncertain, make an estimate, no need to repeat the examples and no other information can be provided. ".format(Text, self.icl)
            # question_1 += "\nExamples: "
            question_e = ""
            for i in range(len(self.examples)):
                question_e += "Molecule: {}".format(self.examples.loc[i, "SMILES"])
                question_e += "\n\n{}: {}".format(Pred, str(self.examples.loc[i, self.task]))
            # question_2 = "\n\nQuestion: "
            if self.use_smiles:
                question_2 = "Molecule: {}".format(data["SMILES"])
            else:
                question_2 = "Molecule: "
            post_question = "\n\n{}: ".format(Pred)
        else:
            question_1 = "Given the molecule, answer the following question. {} Please answer the question with a numerical value only, if uncertain, make an estimate, no other information can be provided. ".format(Text)
            question_e = ""
            if self.use_smiles:
                question_2 = "Molecule: {}".format(data["SMILES"])
            else:
                question_2 = "Molecule: "
            post_question = "\n\n{}: ".format(Pred)

        return {
            'id': index,
            'question_1': question_1,
            'question_e': question_e,
            'question_2': question_2,
            'smiles': data["SMILES"],
            'graph': smiles2data(data["SMILES"]),
            'fingerprint': get_fp_vec(data["SMILES"]),
            'post_question': post_question,
            'label': str(data[self.task]),
        }