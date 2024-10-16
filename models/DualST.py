import torch.nn as nn
import torch.nn.functional as F
import torch
from models.ModelConfiguration import ModelConfigurationTaxiBJ, ModelConfigurationBikeNYC
from models.layer.gse import GlobalSemanticEncoder
from models.layer.stnorm import STnorm
from models.layer.conv import LayerNorm,Block
from models.layer.iTransformer import iTransformer
import numpy as np

class TemporalCausalityBlock(nn.Module):

    def __init__(self,dconf,mconf,drop_path_rate=0.2):
        super().__init__()
        self.dconf = dconf
        self.mconf = mconf
        dim = self.mconf.channel_c
        depth = self.mconf.depth_c

        self.stem = nn.Sequential(
            nn.Conv2d(self.dconf.dim_flow, dim, kernel_size=3, padding=1),
            LayerNorm(dim, eps=1e-6, data_format="channels_first"))
        self.conv1 = nn.Sequential(*[Block(dim=dim,) for j in range(depth)])

        self.conv2 = nn.Conv2d(dim, self.dconf.dim_flow, kernel_size=3, stride=1, padding=1)
        self.act1 = nn.GELU()
        self.act2 = nn.GELU()
        self.fc = nn.Linear(in_features=self.dconf.dim_flow * self.dconf.dim_h * self.dconf.dim_w,
                            out_features=self.mconf.transformer_dmodel)

    def forward(self,x,ext=0):
        x = self.stem(x)
        ext = ext.reshape(-1, self.mconf.channel_c , self.dconf.dim_h, self.dconf.dim_w)
        out = x + ext
        out = self.conv1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        out = out.flatten(1)
        out = self.fc(out)
        return out

class TemporalContextualBlock(nn.Module):

    def __init__(self,dconf,mconf):
        super().__init__()
        self.dconf = dconf
        self.mconf = mconf
        dim = self.mconf.channel_p
        depth = self.mconf.depth_p
        self.stem = nn.Sequential(
            nn.Conv2d(self.dconf.dim_flow*(self.dconf.len_period+self.dconf.len_trend), dim, kernel_size=3, padding=1),
            LayerNorm(dim, eps=1e-6, data_format="channels_first"))
        self.conv1 = nn.Sequential(*[Block(dim=dim,) for j in range(depth)])
        self.act1 = nn.GELU()
        self.act2 = nn.GELU()
        self.conv2 = nn.Conv2d(dim, self.dconf.dim_flow, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(self.mconf.channel_c, dim, kernel_size=3, stride=1, padding=1)
    def forward(self,x,ext=0):
        x = self.stem(x)
        ext = ext.reshape(-1, self.mconf.channel_c, self.dconf.dim_h, self.dconf.dim_w)
        ext = self.conv3(ext)
        out = x + ext
        out = self.conv1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        return out

class DualST(nn.Module):

    def __init__(self,dconf):
        super().__init__()
        self.dconf = dconf
        if self.dconf.name == 'TaxiBJ':
            self.mconf = ModelConfigurationTaxiBJ()
        elif self.dconf.name == 'BikeNYC':
            self.mconf = ModelConfigurationBikeNYC()
        else:
            raise ValueError('The data set does not exist')

        self.ext_c = nn.Sequential(
            nn.Linear(self.dconf.ext_dim, self.mconf.ext_cp_channel),
            nn.Dropout(self.mconf.ext_dropout),
            nn.GELU(),
            nn.Linear(self.mconf.ext_cp_channel,
                      self.mconf.channel_c * self.dconf.dim_h * self.dconf.dim_w),
            nn.GELU()
        )

        self.ext_p = nn.Sequential(
            nn.Linear(self.dconf.ext_dim*(self.dconf.len_period+self.dconf.len_trend), self.mconf.ext_cp_channel),
            nn.Dropout(self.mconf.ext_dropout),
            nn.GELU(),
            nn.Linear(self.mconf.ext_cp_channel,
                      self.mconf.channel_c * self.dconf.dim_h * self.dconf.dim_w),
            nn.GELU()
        )

        self.net_c = nn.ModuleList()
        self.net_p = TemporalContextualBlock(self.dconf,self.mconf)
        for i in range(self.dconf.len_close):
            net = TemporalCausalityBlock(self.dconf, self.mconf)
            self.net_c.append(net)


        self.W1 = nn.Parameter(torch.rand(self.dconf.dim_flow, self.dconf.dim_h,self.dconf.dim_w))
        self.W2 = nn.Parameter(torch.rand(self.dconf.dim_flow, self.dconf.dim_h, self.dconf.dim_w))

        self.pre_token = nn.Parameter(torch.randn(1, self.mconf.transformer_dmodel))

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.mconf.transformer_dmodel,
                                                   nhead=self.mconf.transformer_nhead,
                                                   dropout=self.mconf.transformer_dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, self.mconf.transformer_nlayers)

        self.temporal_norm = STnorm(mean_lr=1e-6, gate_lr=0.001, scale_lr=10)

        self.FC = nn.Linear(in_features=self.mconf.transformer_dmodel * (self.dconf.len_close),
                            out_features=self.dconf.dim_flow * self.dconf.dim_h * self.dconf.dim_w)

        HW_tuple = (self.dconf.dim_h, self.dconf.dim_w)
        self.gse_c = GlobalSemanticEncoder(input_size=HW_tuple, in_chans=self.dconf.dim_flow*self.dconf.len_close)
        self.gse_p = GlobalSemanticEncoder(input_size=HW_tuple, in_chans=self.dconf.dim_flow*(self.dconf.len_period+self.dconf.len_trend))

        self.ext_l = nn.Sequential(
            nn.Linear(self.dconf.ext_dim, self.mconf.extnn_l_channel),
            nn.Dropout(self.mconf.ext_dropout),
            nn.GELU(),
            nn.Linear(self.mconf.extnn_l_channel, self.dconf.dim_flow * self.dconf.dim_h * self.dconf.dim_w),
            nn.GELU()
        )

        self.W1_group = nn.Parameter(torch.rand(self.mconf.transformer_dmodel))
        self.W2_group = nn.Parameter(torch.rand(self.mconf.transformer_dmodel))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.04))

        self.group_conv = nn.Sequential(
            nn.Linear(self.mconf.transformer_dmodel, self.dconf.dim_flow * self.dconf.dim_h * self.dconf.dim_w),
            nn.Dropout(0.2),
            nn.GELU(),
        )

    def forward(self,X, X_ext, Y_ext):
        B,_,H,W = X.shape
        inputs_c = X[:,:self.dconf.len_close*self.dconf.dim_flow,:,:]
        inputs_p = X[:,self.dconf.len_close*self.dconf.dim_flow:, :, :]
        X_ext_c = X_ext[:,:self.dconf.len_close,:]
        X_ext_p = X_ext[:,self.dconf.len_close:,:]

        gse_output_c = self.gse_c(inputs_c,return_attn=True)[0]
        gse_output_p = self.gse_p(inputs_p,return_attn=True)[0]

        logit_scale = self.logit_scale.exp()
        logits_tp = logit_scale * gse_output_c @ gse_output_p.t()
        logits_pt = logits_tp.t()

        gse_output_p = self.group_conv(gse_output_p)
        gse_output_p = gse_output_p.reshape(-1, self.dconf.dim_flow, self.dconf.dim_h, self.dconf.dim_w)

        inputs_c = torch.split(inputs_c, self.dconf.dim_flow, 1)

        ext_outputs_c = self.ext_c(X_ext_c)
        ext_outputs_p = self.ext_p(X_ext_p.reshape(-1,self.dconf.ext_dim*(self.dconf.len_period+self.dconf.len_trend)))
        E_ems_c = torch.split(ext_outputs_c, 1, 1)

        transformer_inputs = []
        for i in range(self.dconf.len_close):
            X_em = self.net_c[i](inputs_c[i], E_ems_c[i].squeeze(1))
            transformer_inputs.append(X_em)

        transformer_inputs = torch.stack(transformer_inputs, 0)
        transformer_inputs = transformer_inputs + gse_output_c
        transformer_inputs = self.temporal_norm(transformer_inputs.permute(1, 2, 0))
        transformer_inputs = transformer_inputs.permute(2,0,1)

        res_temporal = self.transformer_encoder(transformer_inputs)
        transformer_outputs = res_temporal + transformer_inputs
        transformer_outputs = transformer_outputs.transpose(0,1)
        
        out = torch.flatten(transformer_outputs, 1)
        out = self.FC(out)
        main_out_c = out.reshape(-1, self.dconf.dim_flow, self.dconf.dim_h, self.dconf.dim_w)

        main_out_p = self.net_p(inputs_p,ext_outputs_p)
        main_out_p = main_out_p + gse_output_p

        main_out = self.W1*main_out_c + self.W2*main_out_p
        ext_out = self.ext_l(Y_ext)
        ext_out = ext_out.reshape(-1, self.dconf.dim_flow, self.dconf.dim_h, self.dconf.dim_w)

        main_out = main_out + ext_out

        main_out = torch.tanh(main_out)
        return main_out ,logits_tp,logits_pt

    def load(self, file_path):
        self.load_state_dict(torch.load(file_path))
        print("The training model was successfully loaded.")

