import itertools
Z = 20
# Number of times that each amino acid is present in the binding pocket of a catecholamine in the available crystals
# See PDB folder
PDB_frequencies = {'ASP': 168, 'ARG': 111, 'LYS': 63, 'HIP': 114, 'GLU': 125, 'GLN': 119, 'SER': 133, 'ASN': 105, 'THR': 99,
                   'PHE': 186, 'TRP': 52, 'TYR': 187, 'ALA': 161, 'LEU': 184, 'ILE': 107, 'MET': 70, 'VAL': 138, 'GLY': 174, 'PRO': 95, 'CYS': 32}


AA = ['ALA', 'ASP', 'GLN', 'GLY', 'ILE', 'LEU', 'PHE', 'PRO', 'SER', 'TRP', 'TYR']  # nocharged (and ASP)
charges = {'ARG': 1, 'LYS': 1, 'HIP': 1, 'ASP': -1, 'GLU': -1}
A3 = list(itertools.product(AA, repeat=3))  # produces all possible combinations of 3 amino acids
AA_pairs = []
for a1, a2, a3 in A3:
    Q = charges.get(a1, 0) + charges.get(a2, 0) + charges.get(a3, 0)
    if Q == -1 and a3 == "ASP":
        # only keeps triplets with total charge -1 and an ASP in the last position
        AA_pairs.append((a1, a2))
AA_ordered = ['Gln', 'Ser', 'Tyr', 'Trp', 'Phe', 'Ile', 'Leu', 'Ala',
              'Pro', 'Gly']  # lower case (just for figure labels later on). My future self has no idea what did my older self "order" this amino acids based on
