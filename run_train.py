import numpy as np
from main import main
class Args:
  def __init__(self):
    self.lr = 2e-4
    self.lr_backbone_names = ["backbone.0"]
    self.lr_backbone = 2e-5
    self.lr_linear_proj_names = ['reference_points', 'sampling_offsets']
    self.lr_linear_proj_mult = 0.1
    self.batch_size = 1
    self.weight_decay = 1e-4
    self.epochs = 50
    self.lr_drop = 40
    self.lr_drop_epochs = None
    self.clip_max_norm = 0.1

    self.sgd = True

    # Variants of Deformable DETR
    self.with_box_refine = False
    self.two_stage = False

    # Model parameters
    self.frozen_weights = None

    # * Backbone
    self.backbone = 'resnet50'
    self.dilation = True
    self.position_embedding = 'sine'
    self.position_embedding_scale =  2 * np.pi
    self.num_feature_levels = 4
    
    # * Transformer
    self.enc_layers = 6
    self.dec_layers = 6
    self.dim_feedforward = 1024
    self.hidden_dim = 256
    self.dropout = 0.1
    self.nheads = 8
    self.num_queries = 300
    self.dec_n_points = 4
    self.enc_n_points = 4

    # * Segmentation
    self.masks = False

    # Loss
    self.aux_loss = False

    # * Matcher
    self.set_cost_class = 2
    self.set_cost_bbox = 5
    self.set_cost_giou = 2

    # * Loss coefficients
    self.mask_loss_coef = 1
    self.dice_loss_coef = 1
    self.cls_loss_coef = 2
    self.bbox_loss_coef = 5
    self.giou_loss_coef = 2
    self.focal_alpha = 0.25

    # dataset parameters
    self.dataset_file = 'hod'
    self.dataset_path = './data/hod'
    self.train_anns = 'hod_anns_coco_train.json'
    self.val_anns = 'hod_anns_coco_test.json'
    self.remove_difficult = True
    self.num_classes = 2

    self.output_dir = 'checkpoints'
    self.device = 'cuda'
    self.seed = 42
    self.resume = ''
    #self.resume = 'checkpoints/r50_deformable_detr-checkpoint.pth'
    self.start_epoch = 0
    self.eval = False
    self.num_workers = 2
    self.cache_mode = False

if __name__ == '__main__':
  args = Args()
  main(args)