"""Training diffusion model on rna inverse design with given split."""

import ml_collections
import torch

def get_config():
    config = ml_collections.ConfigDict()

    # Misc config
    config.exp_type = 'vpsde'
    config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    config.seed = 42
    config.save = True

    # Data config
    config.data = data = ml_collections.ConfigDict()
    data.seq_centered = True
    data.radius = 4.5
    data.top_k = 10
    data.num_rbf = 16
    data.num_posenc = 16
    data.num_conformers = 1
    data.add_noise = -1.0
    data.knn_num = 10

    # SDE
    config.sde = sde = ml_collections.ConfigDict()
    sde.schedule = 'cosine'  # 'linear', 'cosine'
    sde.continuous_beta_0 = 0.1
    sde.continuous_beta_1 = 20.

    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.method = 'ancestral'
    ## set smaller for faster eval
    sampling.steps = 200

    # Model config
    config.model = model = ml_collections.ConfigDict()
    model.geometric_data_parallel = False
    model.ema_decay = 0.999
    model.pred_data = True
    model.self_cond = True
    model.name = 'GVPTransCond'
    model.node_in_dim = (8, 4)
    model.node_h_dim = (512, 128)
    model.edge_in_dim = (32, 1)
    model.edge_h_dim = (128, 1)
    model.num_layers = 4
    model.drop_rate = 0.1
    model.out_dim = 4
    model.time_cond = True
    model.dihedral_angle = True
    model.num_trans_layer = 8
    model.drop_struct = -1.

    model.trans = trans = ml_collections.ConfigDict()
    trans.encoder_embed_dim = 512
    trans.encoder_attention_heads = 16
    trans.attention_dropout = 0.1
    trans.dropout = 0.1
    trans.encoder_ffn_embed_dim = 1024

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = 'AdamW'
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 20000
    optim.grad_clip = 20.
    optim.disable_grad_log = True

    # Evaluation config
    config.eval = eval = ml_collections.ConfigDict()
    eval.model_path = ''
    eval.test_perplexity = False
    eval.test_recovery = True
    eval.n_samples = 1
    eval.sampling_steps = 50
    eval.cond_scale = -1.
    eval.dynamic_threshold = False
    eval.dynamic_thresholding_percentile = 0.95

    return config