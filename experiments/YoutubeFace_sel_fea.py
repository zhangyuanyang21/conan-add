from .default import Experiment

YoutubeFace_sel_fea = Experiment(
    arch='mlp',
    hidden_dim=256,
    verbose=True,
    log_dir='./logs/YoutubeFace_sel_fea/simclr',
    device='cuda',
    extra_record=True,
    opt='adam',
    epochs=100,
    lr=1e-3,
    batch_size=256,
    cluster_hidden_dim=100,
    ds_name='YoutubeFace_sel_fea',
    input_channels=[64, 512, 64, 647, 838],
    views=5,
    clustering_loss_type='ddc',
    num_cluster=31,
    fusion_act='relu',
    use_bn=True,
    contrastive_type='simclr',
    projection_layers=2,
    projection_dim=256,
    prediction_hidden_dim=0,  # Just for simsiam.
    contrastive_lambda=0.01,
    temperature=0.1,
    seed=0,
)
