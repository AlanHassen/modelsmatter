import os
from abc import abstractmethod, ABC

import numpy

from aizynthfinder.chem import TreeMolecule

class GeneralSmilesBasedModel(ABC):
    """

    """

    def __init__(self, model):
        # in case you need to initialize the model, overrite the init and call super()
        # model.model_setup(args)
        self.model = model

    def predict(self, mol: TreeMolecule):
        smiles = mol.smiles
        # model call requires a list of smiles
        reactants, priors = self.model.model_call([smiles])
        # the model returns a nested list, flatten it
        reactants = numpy.array(reactants).flatten()
        priors = numpy.array(priors).flatten()
        return reactants, priors
