import itertools
Z=20
PDB_frequencies = {'ASP': 168, 'ARG': 111, 'LYS': 63, 'HIP': 114, 'GLU': 125, 'GLN': 119, 'SER': 133, 'ASN': 105, 'THR': 99, 'PHE': 186, 'TRP': 52, 'TYR': 187, 'ALA': 161, 'LEU': 184, 'ILE': 107, 'MET': 70, 'VAL': 138, 'GLY': 174, 'PRO': 95, 'CYS': 32}

AA = ['ALA', 'ARG', 'ASP', 'GLN', 'GLY', 'HIP', 'ILE', 'LEU', 'LYS', 'PHE', 'PRO', 'SER', 'TRP', 'TYR']
AA = ['ALA', 'ASP', 'GLN', 'GLY', 'ILE', 'LEU', 'PHE', 'PRO', 'SER', 'TRP', 'TYR'] #nocharged
charges = {'ARG': 1, 'LYS': 1, 'HIP':1, 'ASP':-1, 'GLU':-1}
A3 = list(itertools.product(AA, repeat=3))
AA_pairs = []
for a1, a2, a3 in A3:
    Q = charges.get(a1, 0) + charges.get(a2, 0) + charges.get(a3,0)
    if Q==-1 and a3 == "ASP":
        AA_pairs.append((a1, a2))
#AA_ordered = ['ASP', 'ARG', 'LYS', 'HIP', 'GLN', 'SER', 'TYR', 'TRP', 'PHE', 'ILE', 'LEU', 'ALA', 'PRO', 'GLY']
#AA_ordered = ['GLN', 'SER', 'TYR', 'TRP', 'PHE', 'ILE', 'LEU', 'ALA', 'PRO', 'GLY'] #Excludes charged residues
AA_ordered = ['Gln', 'Ser', 'Tyr', 'Trp', 'Phe', 'Ile', 'Leu', 'Ala', 'Pro', 'Gly'] #lower case
#AA_ordered = ['ALA', 'GLY'] #test