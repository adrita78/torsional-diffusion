from argparse import ArgumentParser

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from rdkit.Chem import AllChem

import pickle
import pandas as pd
from tqdm import tqdm

import yaml
import os
import os.path as osp
import numpy as np
import torch

from utils.utils import get_model
from diffusion.sampling import *


# ============================================================
# Argument parser
# ============================================================

parser = ArgumentParser()

parser.add_argument(
    '--model_dir',
    type=str,
    required=True,
    help='Path to folder with trained model and hyperparameters'
)

parser.add_argument(
    '--ckpt',
    type=str,
    default='best_model.pt',
    help='Checkpoint to use inside the folder'
)

parser.add_argument(
    '--out',
    type=str,
    required=True,
    help='Path to output pickle file'
)

parser.add_argument(
    '--test_csv',
    type=str,
    default='./data/DRUGS/test_smiles.csv',
    help='CSV containing SMILES and number of conformers'
)


# ============================================================
# Molecular initialization
# ============================================================

parser.add_argument(
    '--pre_mmff',
    action='store_true',
    default=False,
    help='Run MMFF on the initial seed conformer'
)

parser.add_argument(
    '--post_mmff',
    action='store_true',
    default=False,
    help='Run MMFF on the final generated conformers'
)

parser.add_argument(
    '--no_random',
    action='store_true',
    default=False,
    help='Do not randomly perturb the seed torsions'
)

parser.add_argument(
    '--no_model',
    action='store_true',
    default=False,
    help='Return seed conformers without running the model'
)

parser.add_argument(
    '--seed_confs',
    default=None,
    help='Path to seed conformers pickle'
)

parser.add_argument(
    '--seed_mols',
    default=None,
    help='Path to seed molecules pickle'
)

parser.add_argument(
    '--single_conf',
    action='store_true',
    default=False,
    help='Start every sample from the same local structure'
)


# ============================================================
# DDDM sampling
# ============================================================

parser.add_argument(
    '--dddm_steps',
    type=int,
    default=20,
    help='Number of DDDM denoising iterations'
)


# ============================================================
# Dataset / generation size
# ============================================================

parser.add_argument(
    '--limit_mols',
    type=int,
    default=None,
    help='Limit number of molecules'
)

parser.add_argument(
    '--confs_per_mol',
    type=int,
    default=None,
    help=(
        'Number of conformers generated per molecule. '
        'If not specified, generate 2x the number in the CSV.'
    )
)


# ============================================================
# Output / runtime
# ============================================================

parser.add_argument(
    '--dump_pymol',
    type=str,
    default=None,
    help='Directory for PDB denoising trajectories'
)

parser.add_argument(
    '--tqdm',
    action='store_true',
    default=False,
    help='Show progress bar'
)

parser.add_argument(
    '--batch_size',
    type=int,
    default=32,
    help='Number of conformers processed in parallel'
)


# ============================================================
# Energy / likelihood
# ============================================================

parser.add_argument(
    '--water',
    action='store_true',
    default=False,
    help='Compute xTB energy in water'
)

parser.add_argument(
    '--xtb',
    type=str,
    default=None,
    help='Path to local xTB installation'
)

parser.add_argument(
    '--no_energy',
    action='store_true',
    default=False,
    help='Skip likelihood / energy calculations'
)


# ============================================================
# DDDM model arguments
# ============================================================

parser.add_argument(
    '--condition',
    type=str,
    default=None,
    help=(
        'Optional path to conditioning tensor. '
        'Only needed if the DDDM model is conditional.'
    )
)


args = parser.parse_args()


# ============================================================
# Basic validation
# ============================================================

if args.dddm_steps <= 0:
    raise ValueError(
        f'--dddm_steps must be > 0, got {args.dddm_steps}'
    )


# ============================================================
# Device
# ============================================================

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

batch_size = args.batch_size

print('Device:', device)
print('DDDM sampling steps:', args.dddm_steps)
print('Batch size:', batch_size)


# ============================================================
# RDKit embedding
# ============================================================

def embed_func(mol, numConfs):

    AllChem.EmbedMultipleConfs(
        mol,
        numConfs=numConfs,
        numThreads=5
    )

    return mol


# ============================================================
# Load seed conformers / molecules
# ============================================================

seed_confs = None

if args.seed_confs:

    print(
        'Using local structures from',
        args.seed_confs
    )

    with open(args.seed_confs, 'rb') as f:
        seed_confs = pickle.load(f)

elif args.seed_mols:

    print(
        'Using molecules from',
        args.seed_mols
    )

    with open(args.seed_mols, 'rb') as f:
        seed_confs = pickle.load(f)


# ============================================================
# Load model hyperparameters
# ============================================================

with open(
    f'{args.model_dir}/model_parameters.yml'
) as f:

    model_params = yaml.full_load(f)


# Add YAML parameters to argparse namespace.

args.__dict__.update(model_params)

# Keep command-line batch size.

args.batch_size = batch_size


# ============================================================
# Load DDDM model
# ============================================================

model = None

if not args.no_model:

    print('Loading model...')

    model = get_model(args)

    checkpoint_path = (
        f'{args.model_dir}/{args.ckpt}'
    )

    print(
        'Loading checkpoint:',
        checkpoint_path
    )

    state_dict = torch.load(
        checkpoint_path,
        map_location=torch.device('cpu')
    )

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model = model.to(device)

    model.eval()

    print('Model loaded successfully.')


# ============================================================
# Optional condition
# ============================================================

condition = None

if args.condition is not None:

    print(
        'Loading DDDM condition from',
        args.condition
    )

    condition = torch.load(
        args.condition,
        map_location=device
    )

    if isinstance(condition, np.ndarray):
        condition = torch.from_numpy(condition)

    condition = condition.to(
        device=device,
        dtype=torch.float32
    )


# ============================================================
# Load test molecules
# ============================================================

test_data = pd.read_csv(
    args.test_csv
).values

if args.limit_mols is not None:

    test_data = test_data[
        :args.limit_mols
    ]

print(
    'Number of molecules:',
    len(test_data)
)


# ============================================================
# Progress bar
# ============================================================

if args.tqdm:

    test_data = tqdm(
        enumerate(test_data),
        total=len(test_data)
    )

else:

    test_data = enumerate(test_data)


# ============================================================
# Generate conformers for one molecule
# ============================================================

def sample_confs(
    raw_smi,
    n_confs,
    smi,
    smi_idx
):

    print(
        '\nGenerating:',
        smi
    )

    print(
        'Number of conformers:',
        n_confs
    )

    # --------------------------------------------------------
    # Get molecular seed
    # --------------------------------------------------------

    if args.seed_confs:

        mol, data = get_seed(
            raw_smi,
            seed_confs=seed_confs,
            dataset=args.dataset
        )

    elif args.seed_mols:

        mol, data = get_seed(
            smi,
            seed_confs=seed_confs,
            dataset=args.dataset
        )

        if mol is not None:
            mol.RemoveAllConformers()

    else:

        mol, data = get_seed(
            smi,
            dataset=args.dataset
        )

    if mol is None:

        print(
            'Failed to get seed:',
            smi
        )

        return None

    # --------------------------------------------------------
    # Number of rotatable bonds
    # --------------------------------------------------------

    n_rotable_bonds = int(
        data.edge_mask.sum()
    )

    print(
        'Rotatable bonds:',
        n_rotable_bonds
    )

    # --------------------------------------------------------
    # Generate initial conformers
    # --------------------------------------------------------

    if args.seed_confs:

        conformers, pdb = embed_seeds(
            mol,
            data,
            n_confs,
            single_conf=args.single_conf,
            smi=raw_smi,
            pdb=args.dump_pymol,
            seed_confs=seed_confs
        )

    else:

        conformers, pdb = embed_seeds(
            mol,
            data,
            n_confs,
            single_conf=args.single_conf,
            pdb=args.dump_pymol,
            embed_func=embed_func,
            mmff=args.pre_mmff
        )

    if not conformers:

        print(
            'Failed to embed:',
            smi
        )

        return None

    # --------------------------------------------------------
    # Randomly perturb seed torsions
    # --------------------------------------------------------

    if (
        not args.no_random
        and n_rotable_bonds > 0
    ):

        conformers = perturb_seeds(
            conformers,
            pdb
        )

    # --------------------------------------------------------
    # DDDM sampling
    # --------------------------------------------------------

    if (
        not args.no_model
        and n_rotable_bonds > 0
    ):

        print(
            f'Running DDDM for '
            f'{args.dddm_steps} steps...'
        )

        conformers = sample_dddm(
            conformers=conformers,
            model=model,
            num_steps=args.dddm_steps,
            batch_size=args.batch_size,
            pdb=pdb,
            mol=mol,
            condition=condition,
            model_kwargs=None
        )

    elif not args.no_model:

        print(
            'Molecule has 0 rotatable bonds. '
            'Skipping DDDM sampling.'
        )

    # --------------------------------------------------------
    # PDB output
    # --------------------------------------------------------

    if args.dump_pymol and pdb is not None:

        if not osp.isdir(
            args.dump_pymol
        ):

            os.makedirs(
                args.dump_pymol,
                exist_ok=True
            )

        pdb.write(
            f'{args.dump_pymol}/{smi_idx}.pdb',
            limit_parts=5
        )

    # --------------------------------------------------------
    # Convert PyG conformers to RDKit molecules
    # --------------------------------------------------------

    mols = []

    for conf in conformers:

        generated_mol = pyg_to_mol(
            mol,
            conf,
            mmff=args.post_mmff,
            rmsd=not args.no_energy
        )

        mols.append(
            generated_mol
        )

    # --------------------------------------------------------
    # Energy / likelihood
    # --------------------------------------------------------

    #
    # IMPORTANT:
    #
    # The current DDDM sampler does not calculate dlogp.
    #
    # Therefore likelihood/free-energy calculation from
    # the original score-based sampler should NOT be called
    # unless you explicitly implement DDDM likelihood.
    #

    if not args.no_energy:

        print(
            'WARNING: --no_energy is recommended for the '
            'current DDDM sampler because DDDM does not '
            'populate euclidean_dlogp.'
        )

        try:

            for generated_mol, conf in zip(
                mols,
                conformers
            ):

                populate_likelihood(
                    generated_mol,
                    conf,
                    water=args.water,
                    xtb=args.xtb
                )

        except Exception as e:

            print(
                'Energy/likelihood calculation failed:',
                e
            )

            print(
                'Continuing without energy filtering.'
            )

    # --------------------------------------------------------
    # xTB filtering
    # --------------------------------------------------------

    if args.xtb:

        mols = [
            mol
            for mol in mols
            if hasattr(mol, 'xtb_energy')
            and mol.xtb_energy is not None
        ]

    return mols


# ============================================================
# Main generation loop
# ============================================================

conformer_dict = {}


for smi_idx, (
    raw_smi,
    n_confs,
    smi
) in test_data:

    # --------------------------------------------------------
    # Determine number of conformers
    # --------------------------------------------------------

    if args.confs_per_mol is not None:

        n_generate = args.confs_per_mol

    else:

        n_generate = 2 * int(n_confs)

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    mols = sample_confs(
        raw_smi=raw_smi,
        n_confs=n_generate,
        smi=smi,
        smi_idx=smi_idx
    )

    if not mols:

        print(
            f'Skipping molecule {smi}'
        )

        continue

    # --------------------------------------------------------
    # Print basic information
    # --------------------------------------------------------

    print(
        f'{smi_idx} '
        f'rotable_bonds={mols[0].n_rotable_bonds} '
        f'n_confs={len(mols)} '
        f'dddm_steps={args.dddm_steps} '
        f'{smi}',
        flush=True
    )

    # --------------------------------------------------------
    # Store generated conformers
    # --------------------------------------------------------

    conformer_dict[smi] = mols


# ============================================================
# Save generated conformers
# ============================================================

if args.out:

    output_dir = osp.dirname(
        args.out
    )

    if output_dir:

        os.makedirs(
            output_dir,
            exist_ok=True
        )

    with open(
        args.out,
        'wb'
    ) as f:

        pickle.dump(
            conformer_dict,
            f
        )

    print(
        '\nSaved generated conformers to:',
        args.out
    )


print(
    'Generated conformers for',
    len(conformer_dict),
    'molecules'
)
