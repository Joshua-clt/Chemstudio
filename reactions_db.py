from typing import Any

# ==================================================================
# CATALOGUE DES RÉACTIONS ORGANIQUES (57 TRANSFORMATIONS · 9 FAMILLES)
# Chaque entrée définit :
#   - category    : famille chimique (pour le filtre dans l'interface)
#   - name        : nom affiché
#   - reagents    : réactifs / conditions typiques (aide-mémoire)
#   - n_reactants : nombre de cartes réactifs attendues (1 ou 2)
#   - smarts      : règle de transformation RDKit
# ==================================================================
REACTION_CATALOG: list[dict[str, Any]] = [
    # --- 1. ACYLATIONS & DÉRIVÉS D'ACIDES ---
    {
        "category": "Acylations & Dérivés d'acides",
        "name": "Estérification de Fischer (Acide + Alcool -> Ester)",
        "reagents": "H2SO4 cat. ou SOCl2, reflux",
        "n_reactants": 2,
        "smarts": "[#6:1][C:2](=[O:3])[OX2H1].[#6;!$(C=O):4][OX2H1:5]>>[#6:1][C:2](=[O:3])[O:5][#6:4]",
    },
    {
        "category": "Acylations & Dérivés d'acides",
        "name": "Estérification par chlorure d'acyle (Acyle-Cl + Alcool)",
        "reagents": "Et3N ou Pyridine, DCM",
        "n_reactants": 2,
        "smarts": "[#6:1][C:2](=[O:3])Cl.[#6;!$(C=O):4][OX2H1:5]>>[#6:1][C:2](=[O:3])[O:5][#6:4]",
    },
    {
        "category": "Acylations & Dérivés d'acides",
        "name": "Estérification par anhydride (Anhydride + Alcool/Phénol)",
        "reagents": "DMAP cat., Pyridine ou Et3N",
        "n_reactants": 2,
        "smarts": "[#6:1][C:2](=[O:3])OC(=O)[#6].[#6;!$(C=O):4][OX2H1:5]>>[#6:1][C:2](=[O:3])[O:5][#6:4]",
    },
    {
        "category": "Acylations & Dérivés d'acides",
        "name": "Couplage peptidique / Amidation (Acide + Amine -> Amide)",
        "reagents": "HATU ou EDC/HOBt, DIPEA, DMF",
        "n_reactants": 2,
        "smarts": "[#6:1][C:2](=[O:3])[OX2H1].[NX3;H2,H1;!$(NC=O);!$(NS=O):4]>>[#6:1][C:2](=[O:3])[N:4]",
    },
    {
        "category": "Acylations & Dérivés d'acides",
        "name": "Amidation par chlorure d'acyle (Acyle-Cl + Amine)",
        "reagents": "Et3N ou DIPEA, DCM, 0 °C",
        "n_reactants": 2,
        "smarts": "[#6:1][C:2](=[O:3])Cl.[NX3;H2,H1;!$(NC=O);!$(NS=O):4]>>[#6:1][C:2](=[O:3])[N:4]",
    },
    {
        "category": "Acylations & Dérivés d'acides",
        "name": "Amidation par anhydride (Anhydride + Amine)",
        "reagents": "Et3N, DCM ou AcOH",
        "n_reactants": 2,
        "smarts": "[#6:1][C:2](=[O:3])OC(=O)[#6].[NX3;H2,H1;!$(NC=O);!$(NS=O):4]>>[#6:1][C:2](=[O:3])[N:4]",
    },
    {
        "category": "Acylations & Dérivés d'acides",
        "name": "Formation de sulfonamide (R-SO2Cl + Amine)",
        "reagents": "Et3N ou Pyridine, DCM",
        "n_reactants": 2,
        "smarts": "[#6:1][S:2](=[O:3])(=[O:4])Cl.[NX3;H2,H1;!$(NC=O);!$(NS=O):5]>>[#6:1][S:2](=[O:3])(=[O:4])[N:5]",
    },
    {
        "category": "Acylations & Dérivés d'acides",
        "name": "Formation d'urée (Isocyanate + Amine)",
        "reagents": "DCM ou THF, t.a.",
        "n_reactants": 2,
        "smarts": "[#6:1][N:2]=[C:3]=[O:4].[NX3;H2,H1;!$(NC=O);!$(NS=O):5]>>[#6:1][NH1:2][C:3](=[O:4])[N:5]",
    },
    {
        "category": "Acylations & Dérivés d'acides",
        "name": "Formation de carbamate (Isocyanate + Alcool)",
        "reagents": "Et3N ou DBTDL cat., THF",
        "n_reactants": 2,
        "smarts": "[#6:1][N:2]=[C:3]=[O:4].[#6;!$(C=O):5][OX2H1:6]>>[#6:1][NH1:2][C:3](=[O:4])[O:6][#6:5]",
    },
    {
        "category": "Acylations & Dérivés d'acides",
        "name": "Hydrolyse d'ester en acide carboxylique (Saponification)",
        "reagents": "LiOH ou NaOH, THF/H2O",
        "n_reactants": 1,
        "smarts": "[#6:1][C:2](=[O:3])[OX2][#6;!$(C=O)]>>[#6:1][C:2](=[O:3])O",
    },
    {
        "category": "Acylations & Dérivés d'acides",
        "name": "Hydrolyse de nitrile en acide carboxylique",
        "reagents": "HCl conc. ou NaOH aq., reflux",
        "n_reactants": 1,
        "smarts": "[#6:1][C:2]#[N]>>[#6:1][C:2](=O)O",
    },

    # --- 2. SUBSTITUTIONS NUCLÉOPHILES (SN2 / SNAr) ---
    {
        "category": "Substitutions nucléophiles",
        "name": "Synthèse d'éther de Williamson (Alcool/Phénol + R-X)",
        "reagents": "NaH ou K2CO3, DMF / THF",
        "n_reactants": 2,
        "smarts": "[#6;!$(C=O):1][OX2H1:2].[CX4:3][Cl,Br,I]>>[#6:1][O:2][C:3]",
    },
    {
        "category": "Substitutions nucléophiles",
        "name": "N-Alkylation d'amine (Amine + Halogénure d'alkyle)",
        "reagents": "K2CO3 ou DIPEA, MeCN / DMF",
        "n_reactants": 2,
        "smarts": "[NX3;H2,H1;!$(NC=O);!$(NS=O):1].[CX4:2][Cl,Br,I]>>[N:1][C:2]",
    },
    {
        "category": "Substitutions nucléophiles",
        "name": "Synthèse de thioéther (Thiol + Halogénure d'alkyle)",
        "reagents": "K2CO3 ou Et3N, DMF / MeCN",
        "n_reactants": 2,
        "smarts": "[#6:1][SX2H1:2].[CX4:3][Cl,Br,I]>>[#6:1][S:2][C:3]",
    },
    {
        "category": "Substitutions nucléophiles",
        "name": "Substitution par azoture (Halogénure d'alkyle -> R-N3)",
        "reagents": "NaN3, DMF ou DMSO",
        "n_reactants": 1,
        "smarts": "[CX4:1][Cl,Br,I]>>[C:1]N=[N+]=[N-]",
    },
    {
        "category": "Substitutions nucléophiles",
        "name": "Cyanation SN2 / Kolbe (Halogénure d'alkyle -> Nitrile)",
        "reagents": "NaCN ou KCN, DMSO / EtOH-H2O",
        "n_reactants": 1,
        "smarts": "[CX4:1][Cl,Br,I]>>[C:1]C#N",
    },
    {
        "category": "Substitutions nucléophiles",
        "name": "Substitution nucléophile aromatique SNAr (Ar-F/Cl activé + Amine)",
        "reagents": "DIPEA ou K2CO3, DMSO / DMF, chauffage",
        "n_reactants": 2,
        "smarts": "[c;H0;$(c1ccc([N+](=O)[O-])cc1),$(c1ccccc1[N+](=O)[O-]),$(c1ccc(C#N)cc1),$(c1ccccn1),$(c1ccncc1):1][F,Cl].[NX3;H2,H1;!$(NC=O);!$(NS=O):2]>>[c:1][N:2]",
    },

    # --- 3. COUPLAGES CROISÉS & ORGANOMÉTALLIQUES ---
    {
        "category": "Couplages & Organométalliques",
        "name": "Couplage de Suzuki-Miyaura (Ar-X + Acide/Ester boronique)",
        "reagents": "Pd(PPh3)4 ou Pd(dppf)Cl2, K2CO3, Dioxane/H2O",
        "n_reactants": 2,
        "smarts": "[c:1][Br,I,Cl].[c:2][B]([O])[O]>>[c:1]-[c:2]",
    },
    {
        "category": "Couplages & Organométalliques",
        "name": "Couplage de Sonogashira (Ar-Br/I + Alcyne terminal)",
        "reagents": "PdCl2(PPh3)2, CuI, Et3N, THF",
        "n_reactants": 2,
        "smarts": "[c:1][Br,I].[#6:2][C:3]#[CH1:4]>>[c:1][C:4]#[C:3][#6:2]",
    },
    {
        "category": "Couplages & Organométalliques",
        "name": "Couplage de Heck (Ar-Br/I + Alcène terminal)",
        "reagents": "Pd(OAc)2, P(o-tol)3, Et3N, DMF",
        "n_reactants": 2,
        "smarts": "[c:1][Br,I].[#6:2][CH1:3]=[CH2:4]>>[c:1][CH1:4]=[CH1:3][#6:2]",
    },
    {
        "category": "Couplages & Organométalliques",
        "name": "Couplage de Buchwald-Hartwig (Ar-X + Amine -> Ar-NR2)",
        "reagents": "Pd2(dba)3, BINAP ou XPhos, NaOtBu, Toluène",
        "n_reactants": 2,
        "smarts": "[c:1][Br,I,Cl].[NX3;H2,H1;!$(NC=O);!$(NS=O):2]>>[c:1][N:2]",
    },
    {
        "category": "Couplages & Organométalliques",
        "name": "Couplage de Stille (Ar-Br/I + Organostannane)",
        "reagents": "Pd(PPh3)4, LiCl, DMF ou Toluène",
        "n_reactants": 2,
        "smarts": "[c:1][Br,I].[#6:2][Sn]([#6])([#6])[#6]>>[c:1]-[#6:2]",
    },
    {
        "category": "Couplages & Organométalliques",
        "name": "Addition de Grignard sur aldéhyde/cétone (R-MgX + C=O)",
        "reagents": "THF anhydre, 0 °C puis NH4Cl aq.",
        "n_reactants": 2,
        "smarts": "[#6:1][Mg][Br,Cl,I].[C;X3;!$(C([O,N,S,Cl,Br,F])=O):2]=[O:3]>>[#6:1][C:2][OH1:3]",
    },
    {
        "category": "Couplages & Organométalliques",
        "name": "Couplage d'Ullmann (Ar-Br/I + Phénol -> Éther diarylique)",
        "reagents": "CuI, Cs2CO3, ligand N,N-diméthylglycine, Dioxane",
        "n_reactants": 2,
        "smarts": "[c:1][Br,I].[c:2][OX2H1:3]>>[c:1][O:3][c:2]",
    },

    # --- 4. RÉDUCTIONS ---
    {
        "category": "Réductions",
        "name": "Réduction d'aldéhyde / cétone en alcool",
        "reagents": "NaBH4, MeOH ou LiAlH4, THF",
        "n_reactants": 1,
        "smarts": "[C;X3;!$(C([O,N,S,Cl,Br,F])=O):1]=[O:2]>>[C:1][OH1:2]",
    },
    {
        "category": "Réductions",
        "name": "Réduction d'ester en alcool primaire",
        "reagents": "LiAlH4, THF, 0 °C -> t.a.",
        "n_reactants": 1,
        "smarts": "[#6:1][C:2](=[O:3])[OX2][#6;!$(C=O)]>>[#6:1][CH2:2][OH1:3]",
    },
    {
        "category": "Réductions",
        "name": "Réduction d'acide carboxylique en alcool primaire",
        "reagents": "LiAlH4 ou BH3·THF, THF",
        "n_reactants": 1,
        "smarts": "[#6:1][C:2](=[O:3])[OX2H1]>>[#6:1][CH2:2][OH1:3]",
    },
    {
        "category": "Réductions",
        "name": "Réduction d'amide en amine",
        "reagents": "LiAlH4 ou BH3·THF, reflux",
        "n_reactants": 1,
        "smarts": "[#6:1][C:2](=[O])[NX3:3]>>[#6:1][CH2:2][N:3]",
    },
    {
        "category": "Réductions",
        "name": "Réduction de dérivé nitro en aniline / amine",
        "reagents": "H2, Pd/C ou Fe / NH4Cl, EtOH-H2O",
        "n_reactants": 1,
        "smarts": "[#6:1][N+1:2](=[O])[O-1]>>[#6:1][NH2;+0:2]",
    },
    {
        "category": "Réductions",
        "name": "Réduction de nitrile en amine primaire",
        "reagents": "LiAlH4, THF ou H2, Ni de Raney",
        "n_reactants": 1,
        "smarts": "[#6:1][C:2]#[N:3]>>[#6:1][CH2:2][NH2:3]",
    },
    {
        "category": "Réductions",
        "name": "Réduction d'azoture en amine (Staudinger)",
        "reagents": "PPh3, THF/H2O ou H2, Pd/C",
        "n_reactants": 1,
        "smarts": "[#6:1][N:2]=[N+]=[N-]>>[#6:1][NH2;+0:2]",
    },
    {
        "category": "Réductions",
        "name": "Hydrogénation catalytique d'alcène en alcane",
        "reagents": "H2 (1 atm), Pd/C, MeOH ou EtOAc",
        "n_reactants": 1,
        "smarts": "[CX3:1]=[CX3:2]>>[C:1]-[C:2]",
    },
    {
        "category": "Réductions",
        "name": "Hydrogénation complète d'alcyne en alcane",
        "reagents": "H2, Pd/C, MeOH",
        "n_reactants": 1,
        "smarts": "[CX2:1]#[CX2:2]>>[C:1]-[C:2]",
    },

    # --- 5. OXYDATIONS ---
    {
        "category": "Oxydations",
        "name": "Oxydation d'alcool primaire en aldéhyde (DMP / Swern)",
        "reagents": "Dess-Martin Periodinane ou (COCl)2/DMSO/Et3N",
        "n_reactants": 1,
        "smarts": "[#6:1][CH2:2][OX2H1:3]>>[#6:1][CH1:2]=[O:3]",
    },
    {
        "category": "Oxydations",
        "name": "Oxydation d'alcool primaire en acide carboxylique (Jones)",
        "reagents": "CrO3/H2SO4 (Jones) ou TEMPO/NaClO2",
        "n_reactants": 1,
        "smarts": "[#6:1][CH2:2][OX2H1:3]>>[#6:1][C:2](=[O:3])O",
    },
    {
        "category": "Oxydations",
        "name": "Oxydation d'alcool secondaire en cétone (DMP / Swern / PCC)",
        "reagents": "DMP, DCM ou Swern",
        "n_reactants": 1,
        "smarts": "[#6:1][CH1:2]([OX2H1:3])[#6:4]>>[#6:1][C:2](=[O:3])[#6:4]",
    },
    {
        "category": "Oxydations",
        "name": "Époxydation d'alcène (Prilezhaev)",
        "reagents": "mCPBA, DCM, 0 °C -> t.a.",
        "n_reactants": 1,
        "smarts": "[CX3:1]=[CX3:2]>>[C:1]1-[C:2]-O-1",
    },
    {
        "category": "Oxydations",
        "name": "Oxydation de sulfure en sulfoxyde",
        "reagents": "mCPBA (1 éq., 0 °C) ou NaIO4",
        "n_reactants": 1,
        "smarts": "[#6:1][SX2:2][#6:3]>>[#6:1][S:2](=O)[#6:3]",
    },
    {
        "category": "Oxydations",
        "name": "Oxydation de sulfure en sulfone",
        "reagents": "mCPBA (2.5 éq.) ou Oxone",
        "n_reactants": 1,
        "smarts": "[#6:1][SX2:2][#6:3]>>[#6:1][S:2](=O)(=O)[#6:3]",
    },

    # --- 6. CARBONYLES & OLÉFINATIONS ---
    {
        "category": "Carbonyles & Oléfinations",
        "name": "Amination réductrice (Aldéhyde/Cétone + Amine -> Amine)",
        "reagents": "NaBH(OAc)3 ou NaBH3CN, AcOH, DCE",
        "n_reactants": 2,
        "smarts": "[C;X3;!$(C([O,N,S,Cl,Br,F])=O):1]=[O].[NX3;H2,H1;!$(NC=O);!$(NS=O):2]>>[C:1][N:2]",
    },
    {
        "category": "Carbonyles & Oléfinations",
        "name": "Formation d'imine / Base de Schiff (C=O + Amine I)",
        "reagents": "Tamis moléculaire 4 Å ou MgSO4, EtOH / DCM",
        "n_reactants": 2,
        "smarts": "[C;X3;!$(C([O,N,S,Cl,Br,F])=O):1]=[O].[NX3H2;!$(NC=O);!$(NS=O):2]>>[C:1]=[N:2]",
    },
    {
        "category": "Carbonyles & Oléfinations",
        "name": "Formation d'oxime (Aldéhyde/Cétone -> Oxime)",
        "reagents": "NH2OH·HCl, NaOAc ou Pyridine, EtOH",
        "n_reactants": 1,
        "smarts": "[C;X3;!$(C([O,N,S,Cl,Br,F])=O):1]=[O]>>[C:1]=NO",
    },
    {
        "category": "Carbonyles & Oléfinations",
        "name": "Oléfination de Wittig (Aldéhyde/Cétone + R-CH2X -> Alcène)",
        "reagents": "PPh3 puis n-BuLi ou KOtBu, THF",
        "n_reactants": 2,
        "smarts": "[C;X3;!$(C([O,N,S,Cl,Br,F])=O):1]=[O].[CX4;H2,H3:2][Cl,Br,I]>>[C:1]=[C:2]",
    },
    {
        "category": "Carbonyles & Oléfinations",
        "name": "Condensation aldolique / Claisen-Schmidt (Aldéhyde + Méthylcétone)",
        "reagents": "NaOH aq., EtOH/H2O",
        "n_reactants": 2,
        "smarts": "[#6:1][CH1:2]=[O].[CH3:3][C:4](=[O:5])[#6:6]>>[#6:1][CH1:2]=[CH1:3][C:4](=[O:5])[#6:6]",
    },

    # --- 7. PROTECTIONS & DÉPROTECTIONS ---
    {
        "category": "Protections & Déprotections",
        "name": "Protection Boc d'amine",
        "reagents": "Boc2O, Et3N ou NaHCO3, DCM / THF",
        "n_reactants": 1,
        "smarts": "[NX3;H2,H1;!$(NC=O);!$(NS=O):1]>>[N:1]C(=O)OC(C)(C)C",
    },
    {
        "category": "Protections & Déprotections",
        "name": "Déprotection Boc d'amine",
        "reagents": "TFA, DCM ou HCl 4M dans le dioxane",
        "n_reactants": 1,
        "smarts": "[NX3:1]C(=O)OC(C)(C)C>>[N:1]",
    },
    {
        "category": "Protections & Déprotections",
        "name": "Protection TBS d'alcool / phénol",
        "reagents": "TBSCl, Imidazole, DMF ou DCM",
        "n_reactants": 1,
        "smarts": "[#6;!$(C=O):1][OX2H1:2]>>[#6:1][O:2][Si](C)(C)C(C)(C)C",
    },
    {
        "category": "Protections & Déprotections",
        "name": "Déprotection TBS d'alcool / phénol",
        "reagents": "TBAF (1M THF)",
        "n_reactants": 1,
        "smarts": "[#6:1][OX2:2][Si](C)(C)C(C)(C)C>>[#6:1][OH1:2]",
    },
    {
        "category": "Protections & Déprotections",
        "name": "Débenzylation d'éther benzylique (O-Bn -> OH)",
        "reagents": "H2, Pd/C, MeOH ou EtOH",
        "n_reactants": 1,
        "smarts": "[#6:1][OX2:2]Cc1ccccc1>>[#6:1][OH1:2]",
    },
    {
        "category": "Protections & Déprotections",
        "name": "Déméthylation d'éther méthylique aromatique (Ar-OMe -> Ar-OH)",
        "reagents": "BBr3, DCM, -78 °C -> t.a.",
        "n_reactants": 1,
        "smarts": "[c:1][OX2:2][CH3]>>[c:1][OH1:2]",
    },

    # --- 8. HÉTÉROCYCLES & CYCLOADDITIONS ---
    {
        "category": "Hétérocycles & Cycloadditions",
        "name": "Click Chemistry CuAAC (Alcyne terminal + Azoture -> 1,2,3-Triazole)",
        "reagents": "CuSO4·5H2O, Ascorbate de sodium, t-BuOH/H2O",
        "n_reactants": 2,
        "smarts": "[#6:1][C:2]#[CH1:3].[#6:4][N:5]=[N+:6]=[N-:7]>>[#6:4][N:5]1[C:2]([#6:1])=[CH1:3][N+0:7]=[N+0:6]1",
    },
    {
        "category": "Hétérocycles & Cycloadditions",
        "name": "Cycloaddition de Diels-Alder (1,3-Diène + Diénophile alcène)",
        "reagents": "Chauffage (toluène) ou acide de Lewis",
        "n_reactants": 2,
        "smarts": "[CX3:1]=[CX3:2]-[CX3:3]=[CX3:4].[CX3:5]=[CX3:6]>>[C:1]1-[C:2]=[C:3]-[C:4]-[C:6]-[C:5]-1",
    },
    {
        "category": "Hétérocycles & Cycloadditions",
        "name": "Synthèse de tétrazole (Nitrile -> 1H-Tétrazole)",
        "reagents": "NaN3, NH4Cl ou ZnBr2, DMF, 110 °C",
        "n_reactants": 1,
        "smarts": "[#6:1][C:2]#[N]>>[#6:1][C:2]1=NN=N[NH1]1",
    },
    {
        "category": "Hétérocycles & Cycloadditions",
        "name": "Synthèse de benzimidazole (o-Phénylènediamine + Aldéhyde)",
        "reagents": "Na2S2O5 ou air/AcOH, DMF / EtOH",
        "n_reactants": 2,
        "smarts": "[c:1]([NH2:2]):[c:3]([NH2:4]).[#6:5][CH1:6]=[O]>>[c:1]1:[c:3][N:4]=[C:6]([#6:5])[NH1:2]1",
    },

    # --- 9. SUBSTITUTIONS ÉLECTROPHILES AROMATIQUES (SEAr) ---
    {
        "category": "Substitutions électrophiles (SEAr)",
        "name": "Bromation aromatique (Ar-H -> Ar-Br)",
        "reagents": "Br2/FeBr3 ou NBS, DMF / MeCN",
        "n_reactants": 1,
        "smarts": "[cH1:1]>>[c:1]Br",
    },
    {
        "category": "Substitutions électrophiles (SEAr)",
        "name": "Nitration aromatique (Ar-H -> Ar-NO2)",
        "reagents": "HNO3 / H2SO4, 0 °C",
        "n_reactants": 1,
        "smarts": "[cH1:1]>>[c:1][N+](=O)[O-]",
    },
    {
        "category": "Substitutions électrophiles (SEAr)",
        "name": "Acylation de Friedel-Crafts (Arène + Chlorure d'acyle)",
        "reagents": "AlCl3, DCM ou nitrobenzène",
        "n_reactants": 2,
        "smarts": "[cH1:1].[#6:2][C:3](=[O:4])Cl>>[c:1][C:3](=[O:4])[#6:2]",
    },
]

# Dictionnaire de compatibilité (nom -> SMARTS)
REACTION_TEMPLATES: dict[str, str] = {
    item["name"]: item["smarts"] for item in REACTION_CATALOG
}

# ==================================================================
# RÈGLES DE RÉTROSYNTHÈSE (23 DÉCONNEXIONS STRATÉGIQUES)
# ==================================================================
RETRO_RULES: list[tuple[str, str]] = [
    ("Coupure d'ester (Estérification de Fischer / Schotten-Baumann)", "[#6:1][C:2](=[O:3])-[OX2:4]-[#6;!$(C=O):5]>>[#6:1][C:2](=[O:3])O.[OX2H1:4][#6:5]"),
    ("Coupure d'amide (Couplage peptidique / Chlorure d'acyle)", "[#6:1][C:2](=[O:3])-[NX3;H1,H0:4]-[#6:5]>>[#6:1][C:2](=[O:3])O.[N:4][#6:5]"),
    ("Coupure de sulfonamide (Chlorure de sulfonyle + Amine)", "[#6:1][S:2](=[O:3])(=[O:4])-[NX3;H1,H0:5]-[#6:6]>>[#6:1][S:2](=[O:3])(=[O:4])Cl.[N:5][#6:6]"),
    ("Coupure d'urée (Isocyanate + Amine)", "[#6:1][NH1:2][C:3](=[O:4])[NX3;H1,H0:5][#6:6]>>[#6:1][N:2]=[C:3]=[O:4].[N:5][#6:6]"),
    ("Coupure de carbamate (Isocyanate + Alcool)", "[#6:1][NH1:2][C:3](=[O:4])[OX2:5][#6:6]>>[#6:1][N:2]=[C:3]=[O:4].[OH1:5][#6:6]"),
    ("Couplage de Suzuki-Miyaura (Liaison biaryle Pd0)", "[c:1]-[c:2]>>[c:1]Br.[c:2]B(O)O"),
    ("Couplage de Sonogashira (Liaison Ar-Alcyne)", "[c:1]-[C:2]#[C:3]-[#6:4]>>[c:1]Br.[CH1:2]#[C:3]-[#6:4]"),
    ("Couplage de Heck (Liaison Ar-Alcène)", "[c:1]-[CH1:2]=[CH1:3]-[#6:4]>>[c:1]Br.[CH2:2]=[CH1:3]-[#6:4]"),
    ("Couplage de Buchwald-Hartwig (Liaison Ar-N)", "[c:1]-[NX3;!$(NC=O);!$(NS=O):2]-[#6:3]>>[c:1]Br.[N:2]-[#6:3]"),
    ("Couplage d'Ullmann (Éther diarylique Ar-O-Ar)", "[c:1]-[OX2:2]-[c:3]>>[c:1]Br.[OH1:2][c:3]"),
    ("Synthèse d'éther de Williamson (SN2)", "[CX4:1]-[OX2:2]-[#6;!$(C=O):3]>>[C:1]Br.[OH1:2][#6:3]"),
    ("Synthèse de thioéther (SN2)", "[CX4:1]-[SX2:2]-[#6:3]>>[C:1]Br.[SH1:2][#6:3]"),
    ("N-Alkylation d'amine (SN2)", "[CX4;!$(C=O):1]-[NX3;!$(NC=O);!$(NS=O):2]-[#6:3]>>[C:1]Br.[N:2][#6:3]"),
    ("Amination réductrice (Aldéhyde + Amine)", "[#6:1]-[CH2:2]-[NX3;H1,H0;!$(NC=O);!$(NS=O):3]-[#6:4]>>[#6:1][CH1:2]=O.[N:3][#6:4]"),
    ("Coupure d'imine (Base de Schiff)", "[#6:1]-[CH1:2]=[NX2:3]-[#6:4]>>[#6:1][CH1:2]=O.[NH2:3][#6:4]"),
    ("Addition de Grignard sur carbonyle (Alcool II/III)", "[#6:1]-[CX4;H1,H0:2](-[OX2H1:3])-[#6:4]>>[#6:1][Mg]Br.[C:2](=[O:3])[#6:4]"),
    ("Oléfination de Wittig (Alcène C=C)", "[#6:1]-[CH1:2]=[CH1:3]-[#6:4]>>[#6:1][CH1:2]=O.[#6:4][CH2:3]Br"),
    ("Condensation aldolique / Claisen-Schmidt (Énone)", "[#6:1]-[CH1:2]=[CH1:3]-[C:4](=[O:5])-[#6:6]>>[#6:1][CH1:2]=O.[CH3:3][C:4](=[O:5])[#6:6]"),
    ("Acylation de Friedel-Crafts (Aryl cétone)", "[c:1]-[C:2](=[O:3])-[#6:4]>>[cH1:1].[#6:4][C:2](=[O:3])Cl"),
    ("Cycloaddition Click CuAAC (1,2,3-Triazole)", "[#6:1][c:2]1[cH1:3][n:4][n:5][n:6]1[#6:7]>>[#6:1][C:2]#[CH1:3].[#6:7][N:6]=[N+:5]=[N-:4]"),
    ("Rétro-Diels-Alder (Cyclohexène)", "[CX4;H2,H1:1]1-[CX3:2]=[CX3:3]-[CX4;H2,H1:4]-[CX4;H2,H1:5]-[CX4;H2,H1:6]-1>>[C:1]=[C:2]-[C:3]=[C:4].[C:5]=[C:6]"),
    ("Protection Boc d'amine (Déconnexion Boc2O)", "[#6:1][NX3:2]C(=O)OC(C)(C)C>>[#6:1][N:2].CC(C)(C)OC(=O)OC(=O)OC(C)(C)C"),
    ("Synthèse de benzimidazole (o-Phénylènediamine + Aldéhyde)", "[c:1]1:[c:2][n:3][c:4]([#6:5])[nH1:6]1>>[c:1]([NH2:6]):[c:2]([NH2:3]).[#6:5][CH1:4]=O"),
]