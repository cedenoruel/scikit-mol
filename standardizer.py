#TODO Implement a scikit-learn compatible molecule standardizer

from rdkit import Chem
from sklearn.base import BaseEstimator, TransformerMixin
from rdkit.Chem.MolStandardize import rdMolStandardize

class Standardizer(BaseEstimator, TransformerMixin):
    """ Input a list of rdkit mols, output the same list but standardised 
    """
    def __init__(self, neutralize=True):
        self.neutralize = neutralize
        None

    def transform(self, X):
        arr = []
        for mol in X:
            # Normalizing functional groups
            # https://molvs.readthedocs.io/en/latest/guide/standardize.html
            clean_mol = rdMolStandardize.Cleanup(mol) 
            # Get parents fragments
            parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
            # Neutralise
            if self.neutralize:
                uncharger = rdMolStandardize.Uncharger()
                uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
            # Add to final list
            arr.append(uncharged_parent_clean_mol)
        return(arr)

    def fit(self, X, y=None):
        return self
