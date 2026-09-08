import random
import copy
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

from rdkit import Chem, Geometry
from rdkit.Chem import AllChem

from spyrmsd import molecule, graph

from utils.featurization import featurize_mol, featurize_mol_from_smiles
from utils.torsion import *
from utils.utils import time_limit, TimeoutException
from utils.visualise import PDBFile


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
still_frames = 10


# ============================================================
# MMFF
# ============================================================

def try_mmff(mol):
    try:
        AllChem.MMFFOptimizeMoleculeConfs(
            mol,
            mmffVariant="MMFF94s"
        )
        return True
    except Exception:
        return False


# ============================================================
# Seed generation
# ============================================================

def get_seed(smi, seed_confs=None, dataset="drugs"):

    if seed_confs:

        if smi not in seed_confs:
            print("smile not in seeds", smi)
            return None, None

        mol = seed_confs[smi][0]
        data = featurize_mol(mol, dataset)

    else:

        mol, data = featurize_mol_from_smiles(
            smi,
            dataset=dataset
        )

        if not mol:
            return None, None

    data.edge_mask, data.mask_rotate = get_transformation_mask(data)

    data.edge_mask = torch.tensor(data.edge_mask)

    return mol, data


# ============================================================
# Embed seeds
# ============================================================

def embed_seeds(
    mol,
    data,
    n_confs,
    single_conf=False,
    smi=None,
    embed_func=None,
    seed_confs=None,
    pdb=None,
    mmff=False,
):

    if not seed_confs:

        embed_num_confs = (
            n_confs
            if not single_conf
            else 1
        )

        try:

            mol = embed_func(
                mol,
                embed_num_confs
            )

        except Exception as e:

            print(e)
            pass

        if len(mol.GetConformers()) != embed_num_confs:

            print(
                len(mol.GetConformers()),
                "!=",
                embed_num_confs
            )

            return [], None

        if mmff:
            try_mmff(mol)

    if pdb:
        pdb = PDBFile(mol)

    conformers = []

    for i in range(n_confs):

        data_conf = copy.deepcopy(data)

        if single_conf:

            seed_mol = copy.deepcopy(mol)

        elif seed_confs:

            seed_mol = random.choice(
                seed_confs[smi]
            )

        else:

            seed_mol = copy.deepcopy(mol)

            [
                seed_mol.RemoveConformer(j)
                for j in range(n_confs)
                if j != i
            ]

        data_conf.pos = torch.from_numpy(
            seed_mol
            .GetConformers()[0]
            .GetPositions()
        ).float()

        data_conf.seed_mol = copy.deepcopy(
            seed_mol
        )

        if pdb:

            pdb.add(
                data_conf.pos,
                part=i,
                order=0,
                repeat=still_frames
            )

            if seed_confs:

                pdb.add(
                    data_conf.pos,
                    part=i,
                    order=-2,
                    repeat=still_frames
                )

            pdb.add(
                torch.zeros_like(data_conf.pos),
                part=i,
                order=-1
            )

        conformers.append(data_conf)

    if mol.GetNumConformers() > 1:

        [
            mol.RemoveConformer(j)
            for j in range(n_confs)
            if j != 0
        ]

    return conformers, pdb


# ============================================================
# Perturb initial conformers
# ============================================================

def perturb_seeds(data, pdb=None):

    for i, data_conf in enumerate(data):

        torsion_updates = np.random.uniform(
            low=-np.pi,
            high=np.pi,
            size=data_conf.edge_mask.sum()
        )

        data_conf.pos = modify_conformer(
            data_conf.pos,
            data_conf.edge_index.T[
                data_conf.edge_mask
            ],
            data_conf.mask_rotate,
            torsion_updates
        )

        data_conf.total_perturb = torsion_updates

        if pdb:

            pdb.add(
                data_conf.pos,
                part=i,
                order=1,
                repeat=still_frames
            )

    return data


# ============================================================
# Helper: calculate current torsion angles
# ============================================================

def get_current_torsions(data, device):

    edge_index = data.edge_index
    edge_mask = data.edge_mask

    edge_list = [
        []
        for _ in range(
            int(torch.max(edge_index)) + 1
        )
    ]

    for p in edge_index.T:

        edge_list[p[0]].append(p[1])

    rot_bonds = [
        (p[0], p[1])
        for i, p in enumerate(edge_index.T)
        if edge_mask[i]
    ]

    dihedral = []

    for a, b in rot_bonds:

        # Find neighboring atoms on either side
        a_neighbors = edge_list[a]
        b_neighbors = edge_list[b]

        c = next(
            x for x in a_neighbors
            if x != b
        )

        d = next(
            x for x in b_neighbors
            if x != a
        )

        dihedral.append(
            (
                c.item(),
                a.item(),
                b.item(),
                d.item()
            )
        )

    if len(dihedral) == 0:

        return torch.empty(
            0,
            device=device
        ), None

    dihedral = torch.tensor(
        dihedral,
        dtype=torch.long,
        device=device
    )

    tau = get_torsion_angles(
        dihedral,
        data.pos.to(device),
        1
    )

    return tau.squeeze(0), dihedral


# ============================================================
# DDDM molecular sampler
# ============================================================

def sample_dddm(
    conformers,
    model,
    num_steps=100,
    batch_size=32,
    pdb=None,
    mol=None,
    condition=None,
    model_kwargs=None,
):

    """
    DDDM sampler for molecular conformers.

    IMPORTANT:
        This assumes that `model` has the DDDM forward interface

            model(
                t,
                x_bar,
                batch,
                **model_kwargs
            )

        and returns the direct denoising displacement used by

            x_bar_new =
                x_graph - model(...)

    Unlike the PF-ODE sampler, this function does NOT use

        0.5 * g^2 * eps * score

    and does NOT inject Langevin noise.
    """

    if model_kwargs is None:
        model_kwargs = {}

    conf_dataset = InferenceDataset(
        conformers
    )

    loader = DataLoader(
        conf_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    sampled_conformers = []

    model.eval()

    for batch_idx, batch in enumerate(loader):

        batch = batch.to(device)

        num_graphs = batch.num_graphs

        # ----------------------------------------------------
        # Initial DDDM state
        # ----------------------------------------------------
        #
        # DDDM evolves x_bar at graph level.
        #
        # The dimension must match the input/output dimension
        # expected by your DDDM model.
        #
        # Your current training code uses 74.
        # ----------------------------------------------------

        if condition is None:

            x_bar = torch.zeros(
                num_graphs,
                74,
                device=device,
                dtype=torch.float32
            )

        else:

            x_bar = condition.to(device)

            if x_bar.shape[0] != num_graphs:

                raise ValueError(
                    "condition has incompatible number "
                    "of graphs: "
                    f"{x_bar.shape[0]} != {num_graphs}"
                )

        # ----------------------------------------------------
        # DDDM timestep
        # ----------------------------------------------------
        #
        # Your original code used:
        #
        #     T = self.num_timesteps
        #
        # but if timesteps are indexed into beta arrays,
        # the largest valid index is num_timesteps - 1.
        #
        # We therefore use the actual final timestep index.
        # ----------------------------------------------------

        if hasattr(model, "num_timesteps"):

            T_value = model.num_timesteps - 1

        else:

            T_value = num_steps - 1

        T = torch.full(
            (num_graphs,),
            T_value,
            device=device,
            dtype=torch.long
        )

        # ----------------------------------------------------
        # Initial x
        # ----------------------------------------------------
        #
        # This follows your DDDM implementation:
        #
        #     x_T = randn_like(batch.x)
        #     batch.x = x_T
        #
        # ----------------------------------------------------

        batch.x = torch.randn_like(
            batch.x,
            device=device
        )

        # ----------------------------------------------------
        # DDDM denoising iterations
        # ----------------------------------------------------

        for step in range(num_steps):

            with torch.no_grad():

                # --------------------------------------------
                # Current graph representation
                # --------------------------------------------

                x_graph = global_mean_pool(
                    batch.x,
                    batch.batch
                )

                # --------------------------------------------
                # DDDM direct denoising transition
                # --------------------------------------------

                model_output = model(
                    T,
                    x_bar,
                    batch,
                    **model_kwargs
                )

                x_bar_new = (
                    x_graph - model_output
                )

            # --------------------------------------------
            # Update x_bar
            # --------------------------------------------

            x_bar = x_bar_new

            # --------------------------------------------
            # Broadcast graph state back to nodes
            # --------------------------------------------

            batch.x = x_bar[
                batch.batch
            ]

        # ----------------------------------------------------
        # Final state
        # ----------------------------------------------------

        final_x = x_bar

        # ----------------------------------------------------
        # Convert final graph state back to molecular data
        # ----------------------------------------------------
        #
        # This part depends on what the 74-dimensional DDDM
        # state actually represents.
        #
        # If the 74 dimensions are the molecular coordinates/
        # torsional representation, this must be decoded here.
        #
        # We do NOT automatically interpret the 74 dimensions
        # as torsion angles because that would be mathematically
        # incorrect unless that is how the model was trained.
        # ----------------------------------------------------

        for graph_idx, data_idx in enumerate(
            batch.idx
        ):

            conformer = conformers[
                int(data_idx)
            ]

            conformer.dddm_state = (
                final_x[
                    graph_idx
                ]
                .detach()
                .cpu()
            )

            sampled_conformers.append(
                conformer
            )

        # ----------------------------------------------------
        # PDB output
        # ----------------------------------------------------

        if pdb:

            for conf_idx in range(
                batch.num_graphs
            ):

                coords = batch.pos[
                    batch.ptr[conf_idx]:
                    batch.ptr[conf_idx + 1]
                ]

                pdb.add(
                    coords,
                    part=batch_size * batch_idx + conf_idx,
                    order=num_steps + 2,
                    repeat=still_frames
                )

    return sampled_conformers


# ============================================================
# Original-style sample wrapper
# ============================================================

def sample(
    conformers,
    model,
    sigma_max=np.pi,
    sigma_min=0.01 * np.pi,
    steps=20,
    batch_size=32,
    ode=False,
    likelihood=None,
    pdb=None,

    pg_weight_log_0=None,
    pg_repulsive_weight_log_0=None,
    pg_weight_log_1=None,
    pg_repulsive_weight_log_1=None,
    pg_kernel_size_log_0=None,
    pg_kernel_size_log_1=None,
    pg_langevin_weight_log_0=None,
    pg_langevin_weight_log_1=None,
    pg_invariant=False,
    mol=None,

    # --------------------------------------------------------
    # DDDM arguments
    # --------------------------------------------------------

    sampling_method="dddm",
    dddm_steps=100,
    condition=None,
    model_kwargs=None,
):

    """
    Molecular sampling entry point.

    sampling_method:
        "dddm"     -> DDDM sampler
        "score"    -> original score-based sampler

    The DDDM branch is intentionally separate from the
    original score/PF-ODE branch.
    """

    # ========================================================
    # DDDM
    # ========================================================

    if sampling_method == "dddm":

        return sample_dddm(
            conformers=conformers,
            model=model,
            num_steps=dddm_steps,
            batch_size=batch_size,
            pdb=pdb,
            mol=mol,
            condition=condition,
            model_kwargs=model_kwargs,
        )

    # ========================================================
    # Original score-based sampler
    # ========================================================

    conf_dataset = InferenceDataset(
        conformers
    )

    loader = DataLoader(
        conf_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    sigma_schedule = (
        10 ** np.linspace(
            np.log10(sigma_max),
            np.log10(sigma_min),
            steps + 1
        )[:-1]
    )

    eps = 1 / steps

    # --------------------------------------------------------
    # Prior-guidance graph construction
    # --------------------------------------------------------

    if (
        pg_weight_log_0 is not None
        and pg_weight_log_1 is not None
    ):

        edge_index = conformers[0].edge_index
        edge_mask = conformers[0].edge_mask

        edge_list = [
            []
            for _ in range(
                int(torch.max(edge_index)) + 1
            )
        ]

        for p in edge_index.T:

            edge_list[p[0]].append(
                p[1]
            )

        rot_bonds = [
            (p[0], p[1])
            for i, p in enumerate(edge_index.T)
            if edge_mask[i]
        ]

        dihedral = []

        for a, b in rot_bonds:

            c = (
                edge_list[a][0]
                if edge_list[a][0] != b
                else edge_list[a][1]
            )

            d = (
                edge_list[b][0]
                if edge_list[b][0] != a
                else edge_list[b][1]
            )

            dihedral.append(
                (
                    c.item(),
                    a.item(),
                    b.item(),
                    d.item()
                )
            )

        dihedral_numpy = np.asarray(
            dihedral
        )

        dihedral = torch.tensor(
            dihedral
        )

        if pg_invariant:

            try:

                with time_limit(10):

                    mol_spyrmsd = (
                        molecule.Molecule
                        .from_rdkit(mol)
                    )

                    aprops = (
                        mol_spyrmsd.atomicnums
                    )

                    am = (
                        mol_spyrmsd.adjacency_matrix
                    )

                    G = graph.graph_from_adjacency_matrix(
                        am,
                        aprops
                    )

                    isomorphisms = (
                        graph.match_graphs(G, G)
                    )

                    isomorphisms = [
                        iso[0]
                        for iso in isomorphisms
                    ]

                    isomorphisms = np.asarray(
                        isomorphisms
                    )

                    dih_iso = (
                        isomorphisms[
                            :,
                            dihedral_numpy
                        ]
                    )

                    dih_iso = np.unique(
                        dih_iso,
                        axis=0
                    )

                    if len(dih_iso) > 32:

                        print(
                            "reduce isomorphisms from",
                            len(dih_iso),
                            "to",
                            32
                        )

                        dih_iso = (
                            dih_iso[
                                np.random.choice(
                                    len(dih_iso),
                                    replace=False,
                                    size=32
                                )
                            ]
                        )

                    else:

                        print(
                            "isomorphisms",
                            len(dih_iso)
                        )

                    dih_iso = torch.from_numpy(
                        dih_iso
                    ).to(device)

            except TimeoutException:

                print(
                    "Timeout generating with "
                    "non invariant kernel"
                )

                pg_invariant = False

    # ========================================================
    # Original score sampler
    # ========================================================

    for batch_idx, data in enumerate(loader):

        dlogp = torch.zeros(
            data.num_graphs
        )

        data_gpu = copy.deepcopy(
            data
        ).to(device)

        for sigma_idx, sigma in enumerate(
            sigma_schedule
        ):

            data_gpu.node_sigma = (
                sigma
                * torch.ones(
                    data.num_nodes,
                    device=device
                )
            )

            with torch.no_grad():

                data_gpu = model(
                    data_gpu
                )

            g = (
                sigma
                * torch.sqrt(
                    torch.tensor(
                        2
                        * np.log(
                            sigma_max
                            / sigma_min
                        )
                    )
                )
            )

            z = torch.normal(
                mean=0,
                std=1,
                size=data_gpu.edge_pred.shape
            )

            score = (
                data_gpu.edge_pred.cpu()
            )

            t = sigma_idx / steps

            pg_weight = (
                10 ** (
                    pg_weight_log_0 * t
                    + pg_weight_log_1 * (1 - t)
                )
                if (
                    pg_weight_log_0 is not None
                    and pg_weight_log_1 is not None
                )
                else 0.0
            )

            pg_repulsive_weight = (
                10 ** (
                    pg_repulsive_weight_log_0 * t
                    + pg_repulsive_weight_log_1 * (1 - t)
                )
                if (
                    pg_repulsive_weight_log_0 is not None
                    and pg_repulsive_weight_log_1 is not None
                )
                else 1.0
            )

            # ------------------------------------------------
            # Original PF-ODE / SDE update
            # ------------------------------------------------

            if ode:

                perturb = (
                    0.5
                    * g ** 2
                    * eps
                    * score
                )

                if likelihood:

                    div = divergence(
                        model,
                        data,
                        data_gpu,
                        method=likelihood
                    )

                    dlogp += (
                        -0.5
                        * g ** 2
                        * eps
                        * div
                    )

            else:

                perturb = (
                    g ** 2
                    * eps
                    * score
                    +
                    g
                    * np.sqrt(eps)
                    * z
                )

            # ------------------------------------------------
            # Prior guidance
            # ------------------------------------------------

            if pg_weight > 0:

                n = data.num_graphs

                if pg_invariant:

                    S, D, _ = dih_iso.shape

                    dih_iso_cat = (
                        dih_iso.reshape(-1, 4)
                    )

                    tau = get_torsion_angles(
                        dih_iso_cat,
                        data_gpu.pos,
                        n
                    )

                    tau_diff = (
                        tau.unsqueeze(1)
                        -
                        tau.unsqueeze(0)
                    )

                    tau_diff = torch.fmod(
                        tau_diff + 3 * np.pi,
                        2 * np.pi
                    ) - np.pi

                    tau_diff = (
                        tau_diff.reshape(
                            n,
                            n,
                            S,
                            D
                        )
                    )

                    tau_matrix = torch.sum(
                        tau_diff ** 2,
                        dim=-1,
                        keepdim=True
                    )

                    tau_matrix, indices = (
                        torch.min(
                            tau_matrix,
                            dim=2
                        )
                    )

                    tau_diff = torch.gather(
                        tau_diff,
                        2,
                        indices.unsqueeze(-1)
                        .repeat(
                            1,
                            1,
                            1,
                            D
                        )
                    ).squeeze(2)

                else:

                    tau = get_torsion_angles(
                        dihedral,
                        data_gpu.pos,
                        n
                    )

                    tau_diff = (
                        tau.unsqueeze(1)
                        -
                        tau.unsqueeze(0)
                    )

                    tau_diff = torch.fmod(
                        tau_diff + 3 * np.pi,
                        2 * np.pi
                    ) - np.pi

                    assert torch.all(
                        tau_diff < np.pi + 0.1
                    )

                    assert torch.all(
                        tau_diff > -np.pi - 0.1
                    )

                    tau_matrix = torch.sum(
                        tau_diff ** 2,
                        dim=-1,
                        keepdim=True
                    )

                kernel_size = (
                    10 ** (
                        pg_kernel_size_log_0 * t
                        +
                        pg_kernel_size_log_1 * (1 - t)
                    )
                    if (
                        pg_kernel_size_log_0 is not None
                        and pg_kernel_size_log_1 is not None
                    )
                    else 1.0
                )

                langevin_weight = (
                    10 ** (
                        pg_langevin_weight_log_0 * t
                        +
                        pg_langevin_weight_log_1 * (1 - t)
                    )
                    if (
                        pg_langevin_weight_log_0 is not None
                        and pg_langevin_weight_log_1 is not None
                    )
                    else 1.0
                )

                k = torch.exp(
                    -1 / kernel_size
                    * tau_matrix
                )

                repulsive = (
                    torch.sum(
                        2
                        / kernel_size
                        * tau_diff
                        * k,
                        dim=1
                    )
                    .cpu()
                    .reshape(-1)
                    / n
                )

                perturb = (
                    0.5
                    * g ** 2
                    * eps
                    * score
                )

                perturb += (
                    langevin_weight
                    * (
                        0.5
                        * g ** 2
                        * eps
                        * score
                        +
                        g
                        * np.sqrt(eps)
                        * z
                    )
                )

                perturb += (
                    pg_weight
                    * (
                        g ** 2
                        * eps
                        * (
                            score
                            +
                            pg_repulsive_weight
                            * repulsive
                        )
                    )
                )

            # ------------------------------------------------
            # Apply torsion update
            # ------------------------------------------------

            conf_dataset.apply_torsion_and_update_pos(
                data,
                perturb.numpy()
            )

            data_gpu.pos = (
                data.pos.to(device)
            )

            # ------------------------------------------------
            # PDB
            # ------------------------------------------------

            if pdb:

                for conf_idx in range(
                    data.num_graphs
                ):

                    coords = data.pos[
                        data.ptr[conf_idx]:
                        data.ptr[conf_idx + 1]
                    ]

                    num_frames = (
                        still_frames
                        if sigma_idx == steps - 1
                        else 1
                    )

                    pdb.add(
                        coords,
                        part=(
                            batch_size
                            * batch_idx
                            + conf_idx
                        ),
                        order=sigma_idx + 2,
                        repeat=num_frames
                    )

            # ------------------------------------------------
            # Likelihood
            # ------------------------------------------------

            for i, d in enumerate(
                dlogp
            ):

                conformers[
                    data.idx[i]
                ].dlogp = d.item()

    return conformers


# ============================================================
# Convert PyG data -> RDKit molecule
# ============================================================

def pyg_to_mol(
    mol,
    data,
    mmff=False,
    rmsd=True,
    copy=True
):

    if not mol.GetNumConformers():

        conformer = Chem.Conformer(
            mol.GetNumAtoms()
        )

        mol.AddConformer(
            conformer
        )

    coords = data.pos

    if type(coords) is not np.ndarray:

        coords = (
            coords
            .double()
            .numpy()
        )

    for i in range(
        coords.shape[0]
    ):

        mol.GetConformer(
            0
        ).SetAtomPosition(
            i,
            Geometry.Point3D(
                coords[i, 0],
                coords[i, 1],
                coords[i, 2]
            )
        )

    if mmff:

        try:

            AllChem.MMFFOptimizeMoleculeConfs(
                mol,
                mmffVariant="MMFF94s"
            )

        except Exception:

            pass

    try:

        if rmsd:

            mol.rmsd = AllChem.GetBestRMS(
                Chem.RemoveHs(
                    data.seed_mol
                ),
                Chem.RemoveHs(
                    mol
                )
            )

        mol.total_perturb = (
            data.total_perturb
        )

    except Exception:

        pass

    mol.n_rotable_bonds = (
        data.edge_mask.sum()
    )

    if not copy:
        return mol

    return deepcopy(mol)


# ============================================================
# Dataset
# ============================================================

class InferenceDataset(Dataset):

    def __init__(
        self,
        data_list,
        transform=None
    ):

        super().__init__()

        self.data = data_list
        self.transform = transform

        for i, d in enumerate(
            self.data
        ):

            d.idx = i

    def len(self):

        return len(self.data)

    def get(self, idx):

        data = self.data[idx]

        if self.transform is not None:

            data = self.transform(
                data
            )

        return data

    def apply_torsion_and_update_pos(
        self,
        data,
        torsion_updates
    ):

        pos_new, torsion_updates = (
            perturb_batch(
                data,
                torsion_updates,
                split=True,
                return_updates=True
            )
        )

        for i, idx in enumerate(
            data.idx
        ):

            try:

                self.data[idx].total_perturb += (
                    torsion_updates[i]
                )

            except Exception:

                pass

            self.data[idx].pos = (
                pos_new[i]
            )
