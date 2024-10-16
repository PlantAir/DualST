import torch.nn as nn
import torch.nn.functional as F
import torch
from models.ModelConfiguration import ModelConfigurationTaxiBJ, ModelConfigurationBikeNYC
from models.layer.gse import GlobalSemanticEncoder
from models.layer.stnorm import STnorm
from models.layer.conv import LayerNorm,Block
from models.layer.kan import MoKLayer
import numpy as np

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)

        elif mode == 'denorm':
            x = self._denormalize(x)

        else:
            raise NotImplementedError

        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias

        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean

        return x


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

        self.ext_c = MoKLayer(self.dconf.ext_dim,self.mconf.channel_c * self.mconf.channel_c * self.dconf.dim_h * self.dconf.dim_w)
            # nn.Sequential(
            # nn.Linear(self.dconf.ext_dim, self.mconf.ext_cp_channel),
            # nn.Dropout(self.mconf.ext_dropout),
            # nn.GELU(),
            
            # nn.Linear(self.mconf.ext_cp_channel,
            #           self.mconf.channel_c * self.dconf.dim_h * self.dconf.dim_w),
            # nn.GELU()
        # )

        self.ext_p = MoKLayer(self.dconf.ext_dim*(self.dconf.len_period+self.dconf.len_trend),self.mconf.channel_c * self.dconf.dim_h * self.dconf.dim_w)
            # nn.Sequential(
            # nn.Linear(self.dconf.ext_dim*(self.dconf.len_period+self.dconf.len_trend), self.mconf.ext_cp_channel),
            # nn.Dropout(self.mconf.ext_dropout),
            # nn.GELU(),
            # nn.Linear(self.mconf.ext_cp_channel,
            #           self.mconf.channel_c * self.dconf.dim_h * self.dconf.dim_w),
            # nn.GELU()
        # )

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
        self.rev_close = RevIN(num_features=1024)
        self.rev_period = RevIN(num_features=1024)
        

    def forward(self,X, X_ext, Y_ext):
        B,_,H,W = X.shape
        # print("X.shape",X.shape)
        
        inputs_c = X[:,:self.dconf.len_close*self.dconf.dim_flow,:,:]
        B_c,F_c,H_c,W_c = inputs_c.shape
        
        inputs_c = inputs_c.view(B_c,F_c,H_c*W_c)
        #rint("Inputs_c.shape",inputs_c.shape)
        inputs_c = self.rev_close(inputs_c,mode='norm')
        #print("Inputs_c.shape",inputs_c.shape)
        inputs_c = inputs_c.view(B_c,F_c,H_c,W_c)
        
        inputs_p = X[:,self.dconf.len_close*self.dconf.dim_flow:, :, :]
        B_p,F_p,H_p,W_p = inputs_p.shape
        
        inputs_p = inputs_p.view(B_p,F_p,H_p*W_p)
        inputs_p = self.rev_period(inputs_p,mode='norm')
        #print("Inputs_p.shape",inputs_c.shape)
        inputs_p = inputs_p.view(B_p,F_p,H_p,W_p)


        
        X_ext_c = X_ext[:,:self.dconf.len_close,:]
        X_ext_p = X_ext[:,self.dconf.len_close:,:]

        # print("X_ext_c.shape:",X_ext_c.shape)
        # print("X_ext_p.shape:",X_ext_p.shape)
        # print("X_ext_p.reshape:",X_ext_p.reshape(-1,self.dconf.ext_dim*(self.dconf.len_period+self.dconf.len_trend)).shape)
        gse_output_c = self.gse_c(inputs_c,return_attn=True)[0]
        # gse_output_c = self.rev_close(gse_output_c,mode='denorm')
        gse_output_p = self.gse_p(inputs_p,return_attn=True)[0]
        # gse_output_p = self.rev_period(gse_output_p,mode='denorm')

        logit_scale = self.logit_scale.exp()
        logits_tp = logit_scale * gse_output_c @ gse_output_p.t()
        logits_pt = logits_tp.t()

        gse_output_p = self.group_conv(gse_output_p)
        gse_output_p = gse_output_p.reshape(-1, self.dconf.dim_flow, self.dconf.dim_h, self.dconf.dim_w)

        inputs_c = torch.split(inputs_c, self.dconf.dim_flow, 1)


        # KAN UN SQUEEZE
        X_ext_c.unsqueeze(1)
        ext_outputs_c = self.ext_c(X_ext_c)
        ext_outputs_c.squeeze(1)
        X_ext_p.unsqueeze(1)
        ext_outputs_p = self.ext_p(X_ext_p.reshape(-1,self.dconf.ext_dim*(self.dconf.len_period+self.dconf.len_trend)))
        ext_outputs_p.squeeze(1)


        
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
        #print("main_out_c_original:",main_out_p.shape)
        main_out_p = main_out_p + gse_output_p
        #print("main_out_p_original:",main_out_p.shape)
        
        B_main_c,L_main_c,H_main_c,W_main_c = main_out_c.shape
        B_main_p,L_main_p,H_main_p,W_main_p = main_out_p.shape
        
        main_out_c = main_out_c.view(B_main_c,L_main_c,H_main_c*W_main_c)
        #print("main_out_c_view.shape:",main_out_c.shape)
        main_out_p = main_out_p.view(B_main_p,L_main_p,H_main_p*W_main_p)
        #print("main_out_p_view.shape:",main_out_p.shape)
        
        main_out_c = self.rev_close(main_out_c,mode='denorm')
        main_out_p = self.rev_period(main_out_p,mode='denorm')
        main_out_c = main_out_c.view(B_main_c,L_main_c,H_main_c,W_main_c)
        main_out_p = main_out_p.view(B_main_p,L_main_p,H_main_p,W_main_p)
        
        main_out = self.W1*main_out_c + self.W2*main_out_p
        ext_out = self.ext_l(Y_ext)
        ext_out = ext_out.reshape(-1, self.dconf.dim_flow, self.dconf.dim_h, self.dconf.dim_w)

        main_out = main_out + ext_out

        main_out = torch.tanh(main_out)
        return main_out ,logits_tp,logits_pt

    def load(self, file_path):
        self.load_state_dict(torch.load(file_path))
        print("The training model was successfully loaded.")

