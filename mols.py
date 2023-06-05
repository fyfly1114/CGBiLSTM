from rdkit import Chem

m = Chem.MolFromSmiles('O=C(c1ccc(C=Cc2n[nH]c3ccccc23)cc1)N1CCNCC1')
m = Chem.MolFromSmarts('O=C(c1ccc(C=Cc2n[nH]c3ccccc23)cc1)N1CCNCC1')