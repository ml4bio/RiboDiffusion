import math
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from Bio.PDB.PDBParser import PDBParser
from Bio import SeqIO
import torch
import torch_geometric
import torch_cluster
from scipy.spatial.transform import Rotation
import os
import gc
import pickle
import pdb

NUM_TO_LETTER = np.array(['A', 'G', 'C', 'U'])
LETTER_TO_NUM = {'A': 0, 'G': 1, 'C': 2, 'U': 3}

def get_posenc(edge_index, num_posenc=16):
    # From https://github.com/jingraham/neurips19-graph-protein-design
    num_posenc = num_posenc
    d = edge_index[0] - edge_index[1]

    frequency = torch.exp(
        torch.arange(0, num_posenc, 2, dtype=torch.float32, device=d.device)
        * -(np.log(10000.0) / num_posenc)
    )

    angles = d.unsqueeze(-1) * frequency
    E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    return E


def get_orientations(X):
    # X : num_conf x num_res x 3
    forward = normalize(X[:, 1:] - X[:, :-1])
    backward = normalize(X[:, :-1] - X[:, 1:])
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)


def get_orientations_single(X):
    # X : num_res x 3
    forward = normalize(X[1:] - X[:-1])
    backward = normalize(X[:-1] - X[1:])
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

def get_sidechains(X):
    # X : num_conf x num_res x 3 x 3
    p, origin, n = X[:, :, 0], X[:, :, 1], X[:, :, 2]
    n, p = normalize(n - origin), normalize(p - origin)
    return torch.cat([n.unsqueeze_(-2), p.unsqueeze_(-2)], -2)

def get_sidechains_single(X):
    # X : num_res x 3 x 3
    p, origin, n = X[:, 0], X[:, 1], X[:, 2]
    n, p = normalize(n - origin), normalize(p - origin)
    return torch.cat([n.unsqueeze_(-2), p.unsqueeze_(-2)], -2)

def normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.linalg.norm(tensor, dim=dim, keepdim=True)))


def rbf(D, D_min=0., D_max=20., D_count=16):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].

    TODO switch to DimeNet RBFs
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=D.device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF


@torch.no_grad()
def construct_data_single(coords, seq=None, mask=None, num_posenc=16, num_rbf=16, knn_num=10):
    coords = torch.as_tensor(coords, dtype=torch.float32) # num_res x 3 x 3
    # seq is np.array/string, convert to torch.tensor
    if isinstance(seq, np.ndarray):
        seq = torch.as_tensor(seq, dtype=torch.long)
    else:
        seq = torch.as_tensor(
            [LETTER_TO_NUM[residue] for residue in seq],
            dtype=torch.long
        )

    # Compute features
    # node positions: num_res x 3
    coord_C = coords[:, 1].clone()
    # Construct merged edge index
    edge_index = torch_cluster.knn_graph(coord_C, k=knn_num)
    edge_index = torch_geometric.utils.coalesce(edge_index)

    # Node attributes: num_res x 2 x 3, each
    orientations = get_orientations_single(coord_C)
    sidechains = get_sidechains_single(coords)

    # Edge displacement vectors: num_edges x  3
    edge_vectors = coord_C[edge_index[0]] - coord_C[edge_index[1]]

    # Edge RBF features: num_edges x num_rbf
    edge_rbf = rbf(edge_vectors.norm(dim=-1), D_count=num_rbf)
    # Edge positional encodings: num_edges x num_posenc
    edge_posenc = get_posenc(edge_index, num_posenc)

    node_s = (seq.unsqueeze(-1) == torch.arange(4).unsqueeze(0)).float()
    node_v = torch.cat([orientations, sidechains], dim=-2)
    edge_s = torch.cat([edge_rbf, edge_posenc], dim=-1)
    edge_v = normalize(edge_vectors).unsqueeze(-2)

    node_s, node_v, edge_s, edge_v = map(
        torch.nan_to_num,
        (node_s, node_v, edge_s, edge_v)
    )

    # add mask for invalid residues
    if mask is None:
        mask = coords.sum(dim=(2, 3)) == 0.
    mask = torch.tensor(mask)

    return {'seq': seq,
            'coords': coords,
            'node_s': node_s,
            'node_v': node_v,
            'edge_s': edge_s,
            'edge_v': edge_v,
            'edge_index': edge_index,
            'mask': mask}


# read PDB files directly; modify from ESM
def parse_pdb_direct(pdb_path, temp_save_path=None, chain=None):
    if temp_save_path is not None:
        try:
            if os.path.exists(temp_save_path):
                with open(temp_save_path, 'rb') as f:
                    seq, xyz, mask = pickle.load(f)
                return seq, xyz, mask
        except:
            # pass
            print(f"Error in reading {temp_save_path}, re-generate it.")

    xyz, seq, doubles, min_resn, max_resn = {}, {}, {}, np.inf, -np.inf
    with open(pdb_path, "rb") as f:
        for line in f:
            line = line.decode("utf-8", "ignore").rstrip()

            if line[:6] == "HETATM" and line[17:17 + 3] == "MSE":
                line = line.replace("HETATM", "ATOM  ")
                line = line.replace("MSE", "MET")

            if line[:4] == "ATOM":
                ch = line[21:22]
                if ch == chain or chain is None:
                    atom = line[12:12 + 4].strip()
                    resi = line[17:17 + 3]
                    resi_extended = line[16:17 + 3].strip()
                    resn = line[22:22 + 5].strip()
                    x, y, z = [float(line[i:(i + 8)]) for i in [30, 38, 46]]

                    if resn[-1].isalpha():
                        resa, resn = resn[-1], int(resn[:-1]) - 1
                    else:
                        resa, resn = "", int(resn) - 1
                    if resn < min_resn: min_resn = resn
                    if resn > max_resn: max_resn = resn
                    if resn not in xyz: xyz[resn] = {}
                    if resa not in xyz[resn]: xyz[resn][resa] = {}
                    if resn not in seq: seq[resn] = {}
                    if resa not in seq[resn]:
                        seq[resn][resa] = resi
                    elif seq[resn][resa] != resi_extended:
                        # doubles mark locations in the pdb file where multi residue entries are
                        # present. There's a known bug in TmAlign binary that doesn't read / skip
                        # these entries, so we mark them to create a sequence that is aligned with
                        # gap tokens in such locations.
                        doubles[resn] = True

                    if atom not in xyz[resn][resa]:
                        xyz[resn][resa][atom] = np.array([x, y, z])

    # convert to numpy arrays, fill in missing values
    seq_, xyz_, mask = [], [], []
    for resn in range(min_resn, max_resn + 1):
        ## residue name as seq
        if resn in seq:
            for k in sorted(seq[resn]):
                # seq_.append(aa_3_N.get(seq[resn][k], 20))
                seq_.append(seq[resn][k].strip())
        else:
            # seq_.append(20)
            continue
        ## xyz coordinates [L, 3, 3]
        coords_tmp = np.zeros((3, 3))
        if resn in xyz:
            for k in sorted(xyz[resn]):
                res_name = seq[resn][k].strip()
                if "C4'" in xyz[resn][k]: coords_tmp[0] = xyz[resn][k]["C4'"]
                if "C1'" in xyz[resn][k]: coords_tmp[1] = xyz[resn][k]["C1'"]
                if res_name in ['A', 'G'] and "N9" in xyz[resn][k]: coords_tmp[2] = xyz[resn][k]["N9"]
                if res_name in ['C', 'U'] and "N1" in xyz[resn][k]: coords_tmp[2] = xyz[resn][k]["N1"]
        xyz_.append(coords_tmp)
        mask.append(np.all(coords_tmp != 0.))

    seq_ = ''.join(seq_)
    assert len(seq_) == len(xyz_)
    xyz_ = np.array(xyz_, dtype=np.float32)
    mask = np.array(mask)

    if temp_save_path is not None:
        pickle.dump((seq_, xyz_, mask), open(temp_save_path, 'wb'))
    return seq_, xyz_, mask

def PDBtoData(pdb_path, num_posenc, num_rbf, knn_num):
    seq, coords, mask = parse_pdb_direct(pdb_path)
    return construct_data_single(
        coords,
        seq,
        mask,
        num_posenc=num_posenc,
        num_rbf=num_rbf,
        knn_num=knn_num,
    )

def sample_to_fasta(sample, pdb_name, fasta_path):
    seq = ''.join(list(NUM_TO_LETTER[sample.cpu().numpy()]))
    with open(fasta_path, 'w') as f:
        f.write(f'>{pdb_name}\n')
        f.write(f'{seq}\n')
