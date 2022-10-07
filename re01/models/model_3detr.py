# Copyright (c) Facebook, Inc. and its affiliates.
import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from third_party.pointnet2.pointnet2_modules import PointnetSAModuleVotes
from third_party.pointnet2.pointnet2_utils import furthest_point_sample
from utils.pc_util import scale_points, shift_scale_points

from models.helpers import GenericMLP
from models.position_embedding import PositionEmbeddingCoordsSine
from models.transformer import (MaskedTransformerEncoder, TransformerDecoder,
                                TransformerDecoderLayer, TransformerEncoder,
                                TransformerEncoderLayer)

class PositionalEncoding(nn.Module):

  def __init__(self, d_model, dropout=0.1, max_len=5000):
      super(PositionalEncoding, self).__init__()
      self.dropout = nn.Dropout(p=dropout)

      pe = torch.zeros(max_len, d_model)
      position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
      div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
      pe[:, 0::2] = torch.sin(position * div_term)
      pe[:, 1::2] = torch.cos(position * div_term)
      pe = pe.unsqueeze(0).transpose(0, 1)
      self.register_buffer('pe', pe)

  def forward(self, x):
      x = x + self.pe[:x.size(0), :]
      return self.dropout(x)

class CrossTransformer(nn.Module):
  """
  Cross Transformer layer
  """
  def __init__(self, dropout, d_model = 512, n_head = 4):
    """
    :param dropout: dropout rate
    :param d_model: dimension of hidden state
    :param n_head: number of heads in multi head attention
    """
    super(CrossTransformer, self).__init__()
    self.attention = nn.MultiheadAttention(d_model, n_head, dropout = dropout)

    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)

    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)
    self.dropout3 = nn.Dropout(dropout)

    self.activation = nn.ReLU()

    self.linear1 = nn.Linear(d_model, d_model * 4)
    self.linear2 = nn.Linear(d_model * 4, d_model)

  def forward(self, input1, input2):
    attn_output, attn_weight = self.attention(input1, input2, input2)
    output = input1 + self.dropout1(attn_output)
    output = self.norm1(output)
    ff_output = self.linear2(self.dropout2(self.activation(self.linear1(output))))
    output = output + self.dropout3(ff_output)
    output = self.norm2(output)

    return output


class PCDRETransformer(nn.Module):
  """                                                                                                                                                                                                                                                                            
  Decoder with Transformer.                                                                                                                                                                                                                                                      
  """

  def __init__(self, feature_dim, vocab_size, n_head, n_layers, dropout):
    """                                                                                                                                                                                                                                                                          
    :param n_head: the number of heads in Transformer                                                                                                                                                                                                                            
    :param n_layers: the number of layers of Transformer                                                                                                                                                                                                                         
    """
    super(PCDRETransformer, self).__init__()

    self.feature_dim = feature_dim
    self.embed_dim = feature_dim
    self.vocab_size = vocab_size
    self.dropout = dropout
    self.dropout1 = nn.Dropout(dropout)
                                                                                                                                                                                                                             
    self.vocab_embedding = nn.Embedding(vocab_size, self.embed_dim) #vocaburaly embedding             


    self.d_model = 256
    self.n_layers = n_layers
    

    self.projection = nn.Linear(self.embed_dim, self.d_model) # nn.Conv2d(self.embed_dim, self.d_model, kernel_size = 1)
    self.transformer = nn.ModuleList([CrossTransformer(self.dropout, self.d_model, n_head) for i in range(n_layers)])


    self.init_weights() # initialize some layers with the uniform 
  def init_weights(self):
    self.vocab_embedding.weight.data.uniform_(-0.1,0.1)


  def forward(self, memory, encoded_captions):
    # memory () npoints x batch x channel
    # encoded

    temp_memory = memory.clone()
    batch_size = memory.size(1)
    
    encoded_captions = encoded_captions.view(batch_size,-1)
    tgt = encoded_captions.permute(1,0)
    tgt_embedding = self.vocab_embedding(tgt)
    tgt_embedding = self.dropout1(self.projection(tgt_embedding))                                                                                                                                             

    #print(tgt_embedding.size(0),tgt_embedding.size(1),tgt_embedding.size(2))
    #print(memory.size(0),memory.size(1),memory.size(2))
    
    for l in self.transformer:
      tgt_embedding, memory = l(tgt_embedding, memory), l(memory, tgt_embedding)

    return memory + temp_memory



class DecoderTransformer(nn.Module):
  """
  Decoder with Transformer.
  """

  def __init__(self, feature_dim, vocab_size, n_head, n_layers, dropout):
    """
    :param n_head: the number of heads in Transformer
    :param n_layers: the number of layers of Transformer
    """
    super(DecoderTransformer, self).__init__()

    self.feature_dim = 512
    self.embed_dim = 512
    self.vocab_size = vocab_size
    self.dropout = dropout

    # embedding layer
    self.vocab_embedding = nn.Embedding(vocab_size, self.embed_dim) #vocaburaly embedding
    
    # Transformer layer
    decoder_layer = nn.TransformerDecoderLayer(self.feature_dim, n_head, dim_feedforward = self.feature_dim * 4, dropout=self.dropout)
    self.transformer = nn.TransformerDecoder(decoder_layer, n_layers)
    self.position_encoding = PositionalEncoding(self.feature_dim)
    
    self.projection = nn.Linear(self.feature_dim*4, self.feature_dim) # nn.Conv2d(self.embed_dim, self.d_model, kernel_size = 1)
    
    # Linear layer to find scores over vocabulary
    self.wdc = nn.Linear(self.feature_dim, vocab_size)
    self.dropout = nn.Dropout(p=self.dropout)
    self.init_weights() # initialize some layers with the uniform distribution

  def init_weights(self):
    """
    Initializes some parameters with values from the uniform distribution, for easier convergence
    """
    self.vocab_embedding.weight.data.uniform_(-0.1,0.1)

    self.wdc.bias.data.fill_(0)
    self.wdc.weight.data.uniform_(-0.1,0.1)    
 

  def forward(self, memory, encoded_captions, caption_lengths):
    """
    :param memory: image feature (S, batch, feature_dim)
    :param tgt: target sequence (length, batch)
    :param sentence_index: sentence index of each token in target sequence (length, batch)
    """
    #memory = torch.cat([memory1,memory2],dim=2)

    batch_size = memory.size(1)
    num_queries = memory.size(3)
    #sens_num = memory.size(2)
    
    
    
    memory = memory.permute(3,1,2,0).reshape(num_queries,batch_size,-1) #.reshape(batch_size,-1).unsqueeze(0)   # num_layers, batch, channel, num_queries
    memory = self.dropout(self.projection(memory))
    
    
    #print(memory.size(0),memory.size(1),memory.size(2))
    encoded_captions = encoded_captions.view(batch_size,-1)
    caption_lengths = caption_lengths.view(batch_size,-1)    

    tgt = encoded_captions.permute(1,0)
    tgt_length = tgt.size(0)

    mask = (torch.triu(torch.ones(tgt_length, tgt_length)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.cuda()

    tgt_embedding = self.vocab_embedding(tgt) 
    tgt_embedding = self.position_encoding(tgt_embedding) #(length, batch, feature_dim)
    #print(tgt_embedding.size(0),tgt_embedding.size(1),tgt_embedding.size(2))
    pred = self.transformer(tgt_embedding, memory, tgt_mask = mask) #(length, batch, feature_dim)
    pred = self.wdc(self.dropout(pred)) #(length, batch, vocab_size)

    pred = pred.permute(1,0,2)

    # Sort input data by decreasing lengths
    caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
    encoded_captions = encoded_captions[sort_ind]
    pred = pred[sort_ind]
    decode_lengths = (caption_lengths - 1).tolist()
    over0 = sum(i>0 for i in decode_lengths)    

    return pred[:over0], encoded_captions[:over0], decode_lengths[:over0]


class PlainDecoder(nn.Module):
  """
  Dynamic speaker network.
  """

  def __init__(self, feature_dim, embed_dim, vocab_size, hidden_dim, dropout):
    """
    """
    super(PlainDecoder, self).__init__()

    self.feature_dim = feature_dim
    self.embed_dim = embed_dim
    self.hidden_dim = hidden_dim
    self.vocab_size = vocab_size
    self.dropout = dropout
    self.softmax = nn.Softmax(dim=1) ##### TODO #####

    # embedding layer
    self.embedding = nn.Embedding(vocab_size, embed_dim)
    self.dropout = nn.Dropout(p=self.dropout)

    
    self.decode_step = nn.LSTMCell(embed_dim + feature_dim, hidden_dim, bias=True)

    self.relu = nn.ReLU()
    # Linear layer to find scores over vocabulary
    self.wdc = nn.Linear(hidden_dim, vocab_size)
    self.init_weights() # initialize some layers with the uniform distribution

  def init_weights(self):
    """
    Initializes some parameters with values from the uniform distribution, for easier convergence
    """
    self.embedding.weight.data.uniform_(-0.1,0.1)

    self.wdc.bias.data.fill_(0)
    self.wdc.weight.data.uniform_(-0.1,0.1)    

  def forward(self, l_total, encoded_captions, caption_lengths): # self.captioner(selected_box_features, gt_captions, gt_caplens)# (batch)*channel*8, batch*1*23, batch*1
    ## encoded_captions (batch,sens_num,23), gt_caplens (batch, sens_sum)
    # (num_layers, batch, channel, num_queries), batch*1*23, batch*1
    #num_layers = l_total.size(0)
    l_total = l_total.permute(1,0,2,3)
    batch_size = l_total.size(0)    
    l_total = l_total.view(batch_size,-1)
    
    sens_num = 1
    
    #encoded_captions = encoded_captions.unsqueeze(0).repeat(num_layer,1,1,1)
    #caption_lengths = caption_lengths.unsqueeze(0).repeat(num_layer,1,1)
    encoded_captions = encoded_captions.view(batch_size*sens_num,-1)
    caption_lengths = caption_lengths.view(batch_size*sens_num,-1)
    #l_total = l_total.permute(0,2,1).reshape(batch_size*sens_num,-1)    

    # Sort input data by decreasing lengths
    caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
    l_total = l_total[sort_ind]
    encoded_captions = encoded_captions[sort_ind]

    # Embedding
    embeddings = self.embedding(encoded_captions) # (batch_size, max_caption_length, embed_dim)

    h_ds = torch.zeros(batch_size*sens_num, self.hidden_dim).cuda()
    c_ds = torch.zeros(batch_size*sens_num, self.hidden_dim).cuda()

    decode_lengths = (caption_lengths - 1).tolist()

    over0 = sum(i>0 for i in decode_lengths)

    predictions = torch.zeros(batch_size*sens_num, max(decode_lengths), self.vocab_size).cuda()
    #alphas = torch.zeros(batch_size, max(decode_lengths), 3).cuda() #TODO  ## is three ok?

    

    for t in range(max(decode_lengths)):
      batch_size_t = sum([l > t for l in decode_lengths])
      c_temp = torch.cat([embeddings[:batch_size_t,t,:],l_total[:batch_size_t]], dim = 1)
      
      
      h_ds, c_ds = self.decode_step(c_temp, (h_ds[:batch_size_t], c_ds[:batch_size_t]))

      preds = self.wdc(h_ds) #### TODO #### Dropout?!
      predictions[:batch_size_t, t, :] = preds

    return predictions[:over0], encoded_captions[:over0], decode_lengths[:over0]

class BoxProcessor(object):
    """
    Class to convert 3DETR MLP head outputs into bounding boxes
    """

    def __init__(self, dataset_config):
        self.dataset_config = dataset_config

    def compute_predicted_center(self, center_offset, query_xyz, point_cloud_dims):
        center_unnormalized = query_xyz + center_offset
        center_normalized = shift_scale_points(
            center_unnormalized, src_range=point_cloud_dims
        )
        return center_normalized, center_unnormalized

    def compute_predicted_size(self, size_normalized, point_cloud_dims):
        scene_scale = point_cloud_dims[1] - point_cloud_dims[0]
        scene_scale = torch.clamp(scene_scale, min=1e-1)
        size_unnormalized = scale_points(size_normalized, mult_factor=scene_scale)
        return size_unnormalized

    def compute_predicted_angle(self, angle_logits, angle_residual):
        if angle_logits.shape[-1] == 1:
            # special case for datasets with no rotation angle
            # we still use the predictions so that model outputs are used
            # in the backwards pass (DDP may complain otherwise)
            angle = angle_logits * 0 + angle_residual * 0
            angle = angle.squeeze(-1).clamp(min=0)
        else:
            angle_per_cls = 2 * np.pi / self.dataset_config.num_angle_bin
            pred_angle_class = angle_logits.argmax(dim=-1).detach()
            angle_center = angle_per_cls * pred_angle_class
            angle = angle_center + angle_residual.gather(
                2, pred_angle_class.unsqueeze(-1)
            ).squeeze(-1)
            mask = angle > np.pi
            angle[mask] = angle[mask] - 2 * np.pi
        return angle

    def compute_objectness_and_cls_prob(self, cls_logits):
        assert cls_logits.shape[-1] == self.dataset_config.num_semcls + 1
        cls_prob = torch.nn.functional.softmax(cls_logits, dim=-1)
        objectness_prob = 1 - cls_prob[..., -1]
        return cls_prob[..., :-1], objectness_prob
    
    #def compute_objcls_prob(self, obj_cls_logits): ##Unused now##
    #    objcls_prob = torch.nn.functional.softmax(obj_cls_logits, dim=-1)
    #    return objcls_prob

    def box_parametrization_to_corners(
        self, box_center_unnorm, box_size_unnorm, box_angle
    ):
        return self.dataset_config.box_parametrization_to_corners(
            box_center_unnorm, box_size_unnorm, box_angle
        )


class Model3DETR(nn.Module):
    """
    Main 3DETR model. Consists of the following learnable sub-models
    - pre_encoder: takes raw point cloud, subsamples it and projects into "D" dimensions
                Input is a Nx3 matrix of N point coordinates
                Output is a N'xD matrix of N' point features
    - encoder: series of self-attention blocks to extract point features
                Input is a N'xD matrix of N' point features
                Output is a N''xD matrix of N'' point features.
                N'' = N' for regular encoder; N'' = N'//2 for masked encoder
    - query computation: samples a set of B coordinates from the N'' points
                and outputs a BxD matrix of query features.
    - decoder: series of self-attention and cross-attention blocks to produce BxD box features
                Takes N''xD features from the encoder and BxD query features.
    - mlp_heads: Predicts bounding box parameters and classes from the BxD box features
    """

    def __init__(
        self,
        pre_encoder,
        encoder,
        decoder,
        dataset_config,
        encoder_dim=256,
        decoder_dim=256,
        position_embedding="fourier",
        mlp_dropout=0.3,
        num_queries=256,
    ):
        super().__init__()

        self.pre_encoder = pre_encoder
        self.encoder = encoder
        if hasattr(self.encoder, "masking_radius"):
            hidden_dims = [encoder_dim]
        else:
            hidden_dims = [encoder_dim, encoder_dim]
        self.encoder_to_decoder_projection = GenericMLP(
            input_dim=encoder_dim,
            hidden_dims=hidden_dims,
            output_dim=decoder_dim,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            output_use_activation=True,
            output_use_norm=True,
            output_use_bias=False,
        )
        self.pos_embedding = PositionEmbeddingCoordsSine(
            d_pos=decoder_dim, pos_type=position_embedding, normalize=True
        )
        self.query_projection = GenericMLP(
            input_dim=decoder_dim,
            hidden_dims=[decoder_dim],
            output_dim=decoder_dim,
            use_conv=True,
            output_use_activation=True,
            hidden_use_bias=True,
        )
        self.decoder = decoder
        self.build_mlp_heads(dataset_config, decoder_dim, mlp_dropout)

        self.build_captioner(dataset_config) ##TODO##
        self.build_reer(dataset_config)
        

        self.num_queries = num_queries
        self.box_processor = BoxProcessor(dataset_config)
        
    def build_captioner(self, dataset_config):
    
        ##DecoderTransformer feature_dim, vocab_size, n_head, n_layers, dropout)
        #captioner = PlainDecoder(
        #    feature_dim=128,
        #    embed_dim=512,
        #    vocab_size=dataset_config.vocab_size,
        #    hidden_dim=512,
        #    dropout=0.5)

        captioner = DecoderTransformer(
            feature_dim=256,
            vocab_size=dataset_config.vocab_size,
            n_head=2,
            n_layers=2,
            dropout=0.5)
                
        self.captioner = captioner
        
    def build_reer(self, dataset_config):
    
        ##DecoderTransformer feature_dim, vocab_size, n_head, n_layers, dropout)
        #captioner = PlainDecoder(
        #    feature_dim=128,
        #    embed_dim=512,
        #    vocab_size=dataset_config.vocab_size,
        #    hidden_dim=512,
        #    dropout=0.5)

        reer = PCDRETransformer(
            feature_dim=256,
            vocab_size=dataset_config.vocab_size,
            n_head=2,
            n_layers=2,
            dropout=0.5)
                
        self.reer = reer        

    def build_mlp_heads(self, dataset_config, decoder_dim, mlp_dropout):
        mlp_func = partial(
            GenericMLP,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            hidden_dims=[decoder_dim, decoder_dim],
            dropout=mlp_dropout,
            input_dim=decoder_dim,
        )

        # Semantic class of the box
        # add 1 for background/not-an-object class
        semcls_head = mlp_func(output_dim=dataset_config.num_semcls + 1)
        #objsemcls_head = mlp_func(output_dim=dataset_config.num_objsemcls) ##TODO##
        # geometry of the box
        center_head = mlp_func(output_dim=3)
        size_head = mlp_func(output_dim=3)
        angle_cls_head = mlp_func(output_dim=dataset_config.num_angle_bin)
        angle_reg_head = mlp_func(output_dim=dataset_config.num_angle_bin)

        mlp_heads = [
            ("sem_cls_head", semcls_head),
            #("objsem_cls_head", objsemcls_head),
            ("center_head", center_head),
            ("size_head", size_head),
            ("angle_cls_head", angle_cls_head),
            ("angle_residual_head", angle_reg_head),
        ]
        self.mlp_heads = nn.ModuleDict(mlp_heads)

    def get_query_embeddings(self, encoder_xyz, point_cloud_dims):
        query_inds = furthest_point_sample(encoder_xyz, self.num_queries)
        query_inds = query_inds.long()
        query_xyz = [torch.gather(encoder_xyz[..., x], 1, query_inds) for x in range(3)]
        query_xyz = torch.stack(query_xyz)
        query_xyz = query_xyz.permute(1, 2, 0)

        # Gater op above can be replaced by the three lines below from the pointnet2 codebase
        # xyz_flipped = encoder_xyz.transpose(1, 2).contiguous()
        # query_xyz = gather_operation(xyz_flipped, query_inds.int())
        # query_xyz = query_xyz.transpose(1, 2)
        pos_embed = self.pos_embedding(query_xyz, input_range=point_cloud_dims)
        query_embed = self.query_projection(pos_embed)
        return query_xyz, query_embed

    def _break_up_pc(self, pc):
        # pc may contain color/normals.

        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def run_encoder(self, point_clouds):
        xyz, features = self._break_up_pc(point_clouds)
        pre_enc_xyz, pre_enc_features, pre_enc_inds = self.pre_encoder(xyz, features)
        # xyz: batch x npoints x 3
        # features: batch x channel x npoints
        # inds: batch x npoints

        # nn.MultiHeadAttention in encoder expects npoints x batch x channel features
        pre_enc_features = pre_enc_features.permute(2, 0, 1)

        # xyz points are in batch x npointx channel order
        enc_xyz, enc_features, enc_inds = self.encoder(
            pre_enc_features, xyz=pre_enc_xyz
        )
        if enc_inds is None:
            # encoder does not perform any downsampling
            enc_inds = pre_enc_inds.type(torch.int64)
        else:
            # use gather here to ensure that it works for both FPS and random sampling
            enc_inds = torch.gather(pre_enc_inds, 1, enc_inds.type(torch.int64))
        return enc_xyz, enc_features, enc_inds
    

    def get_box_predictions(self, query_xyz, point_cloud_dims, box_features, gt_captions, gt_caplens):  ##TODO##
        """
        Parameters:
            query_xyz: batch x nqueries x 3 tensor of query XYZ coords
            point_cloud_dims: List of [min, max] dims of point cloud
                              min: batch x 3 tensor of min XYZ coords
                              max: batch x 3 tensor of max XYZ coords
            box_features: num_layers x num_queries x batch x channel
        """
        # box_features change to (num_layers x batch) x channel x num_queries
        box_features = box_features.permute(0, 2, 3, 1)
        num_layers, batch, channel, num_queries = (
            box_features.shape[0],
            box_features.shape[1],
            box_features.shape[2],
            box_features.shape[3],
        )
        
        #Create gt bbox features here #############################################################3
        ###----------------------------------------###TODO
        ## Compute query number here #### Select bbox features here ##
        #selected_query = self.get_bboxids(query_xyz, bboxes_forpbb) # -> (batch, 8)
        #selected_box_features = self.select_box_features(box_features, selected_query) # -> (num_layers*batch)*channel*8
        #gt_bbox_num = selected_query.shape[1] # -> 8

        #objcls_logits = self.mlp_heads["objsem_cls_head"](selected_box_features).transpose(1, 2)
        ###----------------------------------------###TODO  
        #selected_box_features = selected_box_features.view(num_layers,batch,channel,-1)   
        
        captioner_predictions, encoded_captions, decode_lengths = self.captioner(box_features, gt_captions, gt_caplens) # (num_layers, batch, channel, num_queries), batch*1*23, batch*1
        
        box_features = box_features.reshape(num_layers * batch, channel, num_queries)

        
        ###----------------------------------------###TODO                
        # mlp head outputs are (num_layers x batch) x noutput x nqueries, so transpose last two dims
        cls_logits = self.mlp_heads["sem_cls_head"](box_features).transpose(1, 2)

        
        center_offset = (
            self.mlp_heads["center_head"](box_features).sigmoid().transpose(1, 2) - 0.5
        )
        size_normalized = (
            self.mlp_heads["size_head"](box_features).sigmoid().transpose(1, 2)
        )
        angle_logits = self.mlp_heads["angle_cls_head"](box_features).transpose(1, 2)
        angle_residual_normalized = self.mlp_heads["angle_residual_head"](
            box_features
        ).transpose(1, 2)

        # reshape outputs to num_layers x batch x nqueries x noutput
        cls_logits = cls_logits.reshape(num_layers, batch, num_queries, -1)
        
        ###----------------------------------------###TODO
        #objcls_logits = objcls_logits.reshape(num_layers, batch, gt_bbox_num, -1)##############TODO#########################
        ###----------------------------------------###TODO
        
        
        
        center_offset = center_offset.reshape(num_layers, batch, num_queries, -1)
        size_normalized = size_normalized.reshape(num_layers, batch, num_queries, -1)
        angle_logits = angle_logits.reshape(num_layers, batch, num_queries, -1)
        angle_residual_normalized = angle_residual_normalized.reshape(
            num_layers, batch, num_queries, -1
        )
        angle_residual = angle_residual_normalized * (
            np.pi / angle_residual_normalized.shape[-1]
        )

        outputs = []
        for l in range(num_layers):
            # box processor converts outputs so we can get a 3D bounding box
            (
                center_normalized,
                center_unnormalized,
            ) = self.box_processor.compute_predicted_center(
                center_offset[l], query_xyz, point_cloud_dims
            )
            angle_continuous = self.box_processor.compute_predicted_angle(
                angle_logits[l], angle_residual[l]
            )
            size_unnormalized = self.box_processor.compute_predicted_size(
                size_normalized[l], point_cloud_dims
            )
            box_corners = self.box_processor.box_parametrization_to_corners(
                center_unnormalized, size_unnormalized, angle_continuous
            )

            # below are not used in computing loss (only for matching/mAP eval)
            # we compute them with no_grad() so that distributed training does not complain about unused variables
            with torch.no_grad():
                (
                    semcls_prob,
                    objectness_prob,
                ) = self.box_processor.compute_objectness_and_cls_prob(cls_logits[l])

            

            box_prediction = {
                "sem_cls_logits": cls_logits[l],
                ##----------------------------------------------------------------##TODO##
                #"objsem_cls_logits": objcls_logits[l],
                "predict_captions": captioner_predictions,
                "encoded_captions":encoded_captions,
                "decode_lengths": decode_lengths,
                ##----------------------------------------------------------------##TODO##
                "center_normalized": center_normalized.contiguous(),
                "center_unnormalized": center_unnormalized,
                "size_normalized": size_normalized[l],
                "size_unnormalized": size_unnormalized,
                "angle_logits": angle_logits[l],
                "angle_residual": angle_residual[l],
                "angle_residual_normalized": angle_residual_normalized[l],
                "angle_continuous": angle_continuous,
                "objectness_prob": objectness_prob,
                "sem_cls_prob": semcls_prob,
                "box_corners": box_corners,
            }
            outputs.append(box_prediction)

        # intermediate decoder layer outputs are only used during training
        aux_outputs = outputs[:-1]
        outputs = outputs[-1]

        return {
            "outputs": outputs,  # output from last layer of decoder
            "aux_outputs": aux_outputs,  # output from intermediate layers of decoder
        }

    def forward(self, inputs, encoder_only=False):
        point_clouds = inputs["point_clouds"]

        enc_xyz, enc_features, enc_inds = self.run_encoder(point_clouds)
        enc_features = self.encoder_to_decoder_projection(
            enc_features.permute(1, 2, 0)
        ).permute(2, 0, 1)
        # encoder features: npoints x batch x channel
        # encoder xyz: npoints x batch x 3

        if encoder_only:
            # return: batch x npoints x channels
            return enc_xyz, enc_features.transpose(0, 1)

        point_cloud_dims = [
            inputs["point_cloud_dims_min"],
            inputs["point_cloud_dims_max"],
        ]
        query_xyz, query_embed = self.get_query_embeddings(enc_xyz, point_cloud_dims)
        # query_embed: batch x channel x npoint
        enc_pos = self.pos_embedding(enc_xyz, input_range=point_cloud_dims)

        # decoder expects: npoints x batch x channel
        enc_pos = enc_pos.permute(2, 0, 1)
        query_embed = query_embed.permute(2, 0, 1)
        tgt = torch.zeros_like(query_embed)
        ### cross-attention decoder add
        
        enc_features = self.reer(enc_features, inputs["res"])
        
        ###
        box_features = self.decoder(
            tgt, enc_features, query_pos=query_embed, pos=enc_pos
        )[0]

        box_predictions = self.get_box_predictions( ###----------------------------------------###TODO       
            query_xyz, point_cloud_dims, box_features, inputs["gt_captions"], inputs["gt_caption_lens"]
        )
        return box_predictions


def build_preencoder(args):
    mlp_dims = [3 * int(args.use_color), 64, 128, args.enc_dim]
    preencoder = PointnetSAModuleVotes(
        radius=0.2,
        nsample=64,
        npoint=args.preenc_npoints,
        mlp=mlp_dims,
        normalize_xyz=True,
    )
    return preencoder


def build_encoder(args):
    if args.enc_type == "vanilla":
        encoder_layer = TransformerEncoderLayer(
            d_model=args.enc_dim,
            nhead=args.enc_nhead,
            dim_feedforward=args.enc_ffn_dim,
            dropout=args.enc_dropout,
            activation=args.enc_activation,
        )
        encoder = TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=args.enc_nlayers
        )
    elif args.enc_type in ["masked"]:
        encoder_layer = TransformerEncoderLayer(
            d_model=args.enc_dim,
            nhead=args.enc_nhead,
            dim_feedforward=args.enc_ffn_dim,
            dropout=args.enc_dropout,
            activation=args.enc_activation,
        )
        interim_downsampling = PointnetSAModuleVotes(
            radius=0.4,
            nsample=32,
            npoint=args.preenc_npoints // 2,
            mlp=[args.enc_dim, 256, 256, args.enc_dim],
            normalize_xyz=True,
        )
        
        masking_radius = [math.pow(x, 2) for x in [0.4, 0.8, 1.2]]
        encoder = MaskedTransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=3,
            interim_downsampling=interim_downsampling,
            masking_radius=masking_radius,
        )
    else:
        raise ValueError(f"Unknown encoder type {args.enc_type}")
    return encoder


def build_decoder(args):
    decoder_layer = TransformerDecoderLayer(
        d_model=args.dec_dim,
        nhead=args.dec_nhead,
        dim_feedforward=args.dec_ffn_dim,
        dropout=args.dec_dropout,
    )
    decoder = TransformerDecoder(
        decoder_layer, num_layers=args.dec_nlayers, return_intermediate=True
    )
    return decoder

def build_captioner(args):
    decoder_layer = TransformerDecoderLayer(
        d_model=args.dec_dim,
        nhead=args.dec_nhead,
        dim_feedforward=args.dec_ffn_dim,
        dropout=args.dec_dropout,
    )
    decoder = TransformerDecoder(
        decoder_layer, num_layers=args.dec_nlayers, return_intermediate=True
    )
    return decoder

def build_3detr(args, dataset_config):
    pre_encoder = build_preencoder(args)
    encoder = build_encoder(args)
    decoder = build_decoder(args)
    captioner = build_captioner(args)
    #reer = build_reer(args)
    model = Model3DETR(
        pre_encoder,
        encoder,
        decoder,
        dataset_config,
        encoder_dim=args.enc_dim,
        decoder_dim=args.dec_dim,
        mlp_dropout=args.mlp_dropout,
        num_queries=args.nqueries,
    )
    output_processor = BoxProcessor(dataset_config)
    return model, output_processor
