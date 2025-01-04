"""
Contributed by: Ruel Cedeno, @cedenoruel
"""

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import StratifiedGroupKFold, BaseCrossValidator, GroupShuffleSplit
import numpy as np
import pandas as pd



def get_scaffold(smiles: str)-> str:
    """
    If no scaffold (e.g. linear molecule), return the original smiles
    """
    scaffold =  MurckoScaffold.MurckoScaffoldSmilesFromSmiles(smiles,includeChirality=True)
    return scaffold if len(scaffold) != 0 else smiles 


def get_scaffold_groups(X) -> np.array:
    """
    X <- array or list of smiles or Mol object
    Returns the scaffold ID of each molecule as integers 
     
    """
    try:
        X = pd.Series(np.array(X).flatten()) 
    except TypeError:
        print("X input must be an array, list, or series containing SMILES or mol object")
    except Exception as e:
        print(e)

    df = pd.DataFrame()
    df["smiles"] =   X if  not isinstance(X[0], Chem.Mol) else pd.Series(X).apply(Chem.MolToSmiles)
    df["scaff_id"] = df["smiles"].apply(get_scaffold) #possible parallelization with joblib ?
    return pd.factorize(df['scaff_id'])[0]





def get_split_idx(X, y, groups,test_size=0.2,random_state=42,stratify=True):

    """
    Return the indices of train and test sets such that no groups are in common between sets
    X <- array/list of smiles or rdkit mol object
    y <- array/list of numerical value or boolean

    if stratify is True, split is made to avoid highly unbalanced test set (e.g. maintain similar distribution of + or - in both sets)


    Example:

    train_idx, test_idx = get_split_idx( X = data.SMILES, groups = data.scaff_id, y=data.pXC50, test_size=0.2 )

    """
    
    if random_state is not None:
        np.random.seed(random_state)   

    #sanity check

    try:
        X = pd.Series(np.array(X).flatten()) 
    except TypeError:
        print("X input must be an array, list, or series")
    except Exception as e:
        print(e)

    #if len(X) == 0 or len(y) == 0 or len(groups):
    #    raise ValueError("X, y, and groups must not be empty")
    if len(X) != len(y) != len(groups):
        raise ValueError("X, y, and groups must have the same size")
    if test_size <= 0:
        raise ValueError("test_size must be > 0")
    
    if stratify: #stratify using the median as threshold
        y_class  = np.array(y) if len(np.unique(y)) == 2 else np.array(y) >= np.median(y)
        n_splits = int(1/test_size) if test_size <= 0.5 else  int(1/(1-test_size))
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,random_state=random_state)

        best_train_idx, best_test_idx = None, None
        best_diff = float('inf')

        # Find the best split that maintains the ratio of positive to negative class 

        for train_idx, test_idx in splitter.split(X, y_class, groups):
            y_train, y_test = y_class[train_idx], y_class[test_idx]
            train_pos_ratio = y_train.mean()
            test_pos_ratio = y_test.mean()
            ratio_diff = abs(train_pos_ratio - test_pos_ratio)
                
            if ratio_diff < best_diff:
                best_diff = ratio_diff
                best_train_idx, best_test_idx = train_idx, test_idx

    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        for train_idx, test_idx in splitter.split(X, y_class, groups):
            best_train_idx, best_test_idx = train_idx, test_idx
    
        
    return best_train_idx, best_test_idx

def train_test_group_split(X, y,groups,test_size=0.2, random_state=42):

    """
    X <- array/list of smiles or mol
    y <- array/list of numerical value or boolean  

    Example:
    X_train, X_test, y_train, y_test = scaffold_split_idx( X = data.SMILES, y = data.pXC50)

    """

    train_idx, test_idx = get_split_idx(X, y,groups,test_size, random_state)
    X_train, X_test= X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    return X_train, X_test, y_train, y_test



class GroupSplitCV(BaseCrossValidator):

    """
    Scaffold split in cross validation or hyperparameter optimization

    Example:

    scaffold_cv = GroupSplitCV(n_splits=5, n_repeats=4, X= data.ROMol, y= data.y, groups = data.scaffold_ID)

    search = RandomizedSearchCV(optimization_pipe, param_distributions=param_dist, n_iter=25, cv=scaffold_cv)

    search.fit(X_train, y_train)
        
    """


    def __init__(self, n_splits, n_repeats, X, y, groups, test_size=None,random_state=42):
        super().__init__()
        self.n_splits=n_splits
        self.n_repeats = n_repeats
        self.X = X
        self.y = y
        self.groups = groups
        self.test_size = test_size if test_size is not None else 1/self.n_splits
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        np.random.seed(self.random_state)
        for repeat in range(self.n_repeats):
            for split in range(self.n_splits):
                random_state_i = np.random.randint(0, 10000)
                train_idx, test_idx = get_split_idx(self.X, self.y, self.groups, self.test_size, random_state_i)
                yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits * self.n_repeats

