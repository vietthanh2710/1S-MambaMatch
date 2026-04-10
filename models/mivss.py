from __future__ import annotations

import math
import random
from functools import partial
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath

def selective_scan_multi_input(
    us,
    dts,
    As,
    Bs,
    Cs,
    Ds,
    u2s=None,
    dt2s=None,
    B2s=None,
    delta_bias=None,
    delta_softplus=False,
    return_last_state=False,
    chunksize=8,
):
    """
    Ported from the notebook for reproducibility.

    Shapes (same convention as notebook):
      us, dts: (B, G*D, L)
      As: (G*D, N)
      Bs, Cs: (B, G, N, L) or (B, N, L)
      Ds: (G*D,)
    """

    def selective_scan_chunk(us, u2s, dts, dt2s, As, Bs, B2s, Cs, hprefix):
        ts = dts.cumsum(dim=0)
        Ats = torch.einsum("gdn,lbgd->lbgdn", As, ts).exp()
        scale = Ats[-1].detach()
        rAts = Ats / scale
        duts = dts * us
        dtBus = torch.einsum("lbgd,lbgn->lbgdn", duts, Bs)

        if u2s is not None and dt2s is not None and B2s is not None:
            t2s = dt2s.cumsum(dim=0)
            At2s = torch.einsum("gdn,lbgd->lbgdn", As, t2s).exp()
            scale2 = At2s[-1].detach()
            rAt2s = At2s / scale2

            du2ts = dt2s * u2s
            dtBu2s = torch.einsum("lbgd,lbgn->lbgdn", du2ts, B2s)
            hs_tmp = rAts * (dtBus / rAts).cumsum(dim=0) + rAt2s * (dtBu2s / rAt2s).cumsum(dim=0)
        else:
            hs_tmp = rAts * (dtBus / rAts).cumsum(dim=0)

        hs = hs_tmp + Ats * hprefix.unsqueeze(0)
        ys = torch.einsum("lbgn,lbgdn->lbgd", Cs, hs)
        return ys, hs

    inp_dtype = us.dtype
    has_D = Ds is not None

    dts = dts.float()
    if dt2s is not None:
        dt2s = dt2s.float()

    if delta_bias is not None:
        dts = dts + delta_bias.view(1, -1, 1).float()
    if delta_softplus:
        dts = F.softplus(dts)

    if len(Bs.shape) == 3:
        Bs = Bs.unsqueeze(1)
    if len(Cs.shape) == 3:
        Cs = Cs.unsqueeze(1)
    B, G, N, L = Bs.shape

    us = us.view(B, G, -1, L).permute(3, 0, 1, 2).float()
    dts = dts.view(B, G, -1, L).permute(3, 0, 1, 2).float()
    As = As.view(G, -1, N).float()
    Bs = Bs.permute(3, 0, 1, 2).float()
    Cs = Cs.permute(3, 0, 1, 2).float()
    Ds = Ds.view(G, -1).float() if has_D else None
    D = As.shape[1]

    if u2s is not None:
        u2s = u2s.view(B, G, -1, L).permute(3, 0, 1, 2).float()
    if dt2s is not None:
        dt2s = dt2s.view(B, G, -1, L).permute(3, 0, 1, 2).float()
    if B2s is not None:
        if len(B2s.shape) == 3:
            B2s = B2s.unsqueeze(1)
        B2s = B2s.permute(3, 0, 1, 2).float()

    oys = []
    hprefix = us.new_zeros((B, G, D, N), dtype=torch.float32)
    for i in range(0, L - 1, chunksize):
        ys, hs = selective_scan_chunk(
            us[i : i + chunksize],
            None if u2s is None else u2s[i : i + chunksize],
            dts[i : i + chunksize],
            None if dt2s is None else dt2s[i : i + chunksize],
            As,
            Bs[i : i + chunksize],
            None if B2s is None else B2s[i : i + chunksize],
            Cs[i : i + chunksize],
            hprefix,
        )
        oys.append(ys)
        hprefix = hs[-1]

    oys = torch.cat(oys, dim=0)
    if has_D:
        oys = oys + Ds * us
    oys = oys.permute(1, 2, 3, 0).view(B, -1, L)
    oys = oys.to(inp_dtype)
    return oys if not return_last_state else (oys, hprefix.view(B, G * D, N))


DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


class Linear2d(nn.Linear):
    def forward(self, x: torch.Tensor):
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(self.weight.shape)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0, channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class gMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0, channels_first=False):
        super().__init__()
        self.channel_first = channels_first
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, 2 * hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
        x = self.fc2(x * self.act(z))
        x = self.drop(x)
        return x


class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).view(1, -1).repeat(d_inner, 1).contiguous()
        A_log = torch.log(A)
        if copies > 0:
            A_log = A_log[None].repeat(copies, 1, 1).contiguous()
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = D[None].repeat(copies, 1).contiguous()
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    @classmethod
    def init_dt_A_D(cls, d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4):
        dt_projs = [cls.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor) for _ in range(k_group)]
        dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0))
        dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0))
        A_logs = cls.A_log_init(d_state, d_inner, copies=k_group, merge=True)
        Ds = cls.D_init(d_inner, copies=k_group, merge=True)
        return A_logs, Ds, dt_projs_weight, dt_projs_bias

    @classmethod
    def init_dt_A_D_s_msk(cls, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4):
        dt_projs = [cls.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor) for _ in range(k_group)]
        dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0))
        dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0))
        Ds = cls.D_init(d_inner, copies=k_group, merge=True)
        return Ds, dt_projs_weight, dt_projs_bias


class SS2Dv0:
    def __initv0__(
        self,
        d_model=256,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        dropout=0.0,
        seq=False,
        force_fp32=True,
        **kwargs,
    ):
        if "channel_first" in kwargs:
            assert not kwargs["channel_first"]
        act_layer = nn.SiLU
        dt_min = 0.001
        dt_max = 0.1
        dt_init = "random"
        dt_scale = 1.0
        dt_init_floor = 1e-4
        bias = False
        conv_bias = True
        d_conv = 3
        k_group = 4
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()

        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 8) if dt_rank == "auto" else dt_rank

        self.forward = self.forwardv0
        if seq:
            self.forward = partial(self.forwardv0, seq=True)
        if not force_fp32:
            self.forward = partial(self.forwardv0, force_fp32=False)

        self.in_proj = nn.Linear(d_model, d_inner, bias=bias)
        self.in_proj_s = nn.Linear(d_model, d_inner, bias=bias)

        self.act: nn.Module = act_layer()
        self.conv2d = nn.Conv2d(
            in_channels=d_inner,
            out_channels=d_inner,
            groups=d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        self.x_proj = [nn.Linear(d_inner * 2, (dt_rank + d_state * 2), bias=False) for _ in range(k_group)]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
            d_state, dt_rank, d_inner * 2, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4
        )

        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.fc = nn.Linear(d_model, d_model, bias=False)

        self.conv_mask1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, padding="same")
        self.conv_mask2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding="same")
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.act_mask = nn.ReLU()

        self.s_msk_proj = [nn.Linear(64, (3 + 1 * 2), bias=False) for _ in range(k_group)]
        self.s_msk_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.s_msk_proj], dim=0))
        del self.s_msk_proj

        self.Ds_s_msk, self.dt_s_msk_projs_weight, self.dt_s_msk_projs_bias = mamba_init.init_dt_A_D_s_msk(
            dt_rank=3, d_inner=64, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, k_group=4
        )

    def forwardv0(self, q: torch.Tensor, s: torch.Tensor, s_msk: torch.Tensor, seq=False, force_fp32=True, **kwargs):
        channel_correlation = torch.einsum("bhwc,hwk->bck", q, s.squeeze(0))
        P = torch.softmax(self.fc(channel_correlation), dim=-1)

        s_permuted = s.repeat(q.shape[0], 1, 1, 1)
        s = torch.einsum("bcc,bhwc->bhwc", P, s_permuted)

        x_q = self.in_proj(q)
        s = self.in_proj_s(s)

        x_q = x_q.permute(0, 3, 1, 2).contiguous()
        x_q = self.conv2d(x_q)
        x_q = self.act(x_q)

        s = s.permute(0, 3, 1, 2).contiguous()
        s = self.conv2d(s)
        s = self.act(s)

        B, D, H, W = x_q.shape
        L = H * W

        x_q_t = torch.transpose(x_q, dim0=2, dim1=3).contiguous().view(B, D, -1)
        s_t = torch.transpose(s, dim0=2, dim1=3).contiguous().view(B, D, -1)
        x_q = x_q.view(B, D, -1)
        s = s.view(B, D, -1)

        interwoven = torch.stack((s, x_q), dim=-1).view(B, D * 2, -1)
        interwoven_t = torch.stack((s_t, x_q_t), dim=-1).view(B, D * 2, -1)

        x_hwwh = torch.stack([interwoven, interwoven_t], dim=1).view(B, 2, 2 * D, -1)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        D_all, N = self.A_logs.shape
        K, Dk, R = self.dt_projs_weight.shape
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        Bs = Bs.contiguous()
        Cs = Cs.contiguous()

        As = -self.A_logs.float().exp()
        Ds = self.Ds.float()
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        if force_fp32:
            xs = xs.float()
            dts = dts.float()
            Bs = Bs.float()
            Cs = Cs.float()

        # support mask processing
        s_msk = s_msk.permute(0, 3, 1, 2).contiguous()
        s_msk = self.down(self.conv_mask1(s_msk))
        s_msk = self.act_mask(s_msk)
        s_msk = self.down(self.conv_mask2(s_msk))

        Bm, Cm, Hm, Wm = s_msk.shape
        s_msk_hwwh = torch.stack(
            [s_msk.view(Bm, Cm, -1), torch.transpose(s_msk, dim0=2, dim1=3).contiguous().view(Bm, Cm, -1)],
            dim=1,
        ).view(Bm, 2, Cm, -1)
        s_msk = torch.cat([s_msk_hwwh, torch.flip(s_msk_hwwh, dims=[-1])], dim=1)

        s_msk_dbl = torch.einsum("b k d l, k c d -> b k c l", s_msk, self.s_msk_proj_weight)
        dts_s_msk, Bs_s_msk = torch.split(s_msk_dbl, [3, 1], dim=2)
        dts_s_msk = torch.einsum("b k r l, k d r -> b k d l", dts_s_msk, self.dt_s_msk_projs_weight)

        s_msk = s_msk.repeat(B, 1, 1, 1)
        dts_s_msk = dts_s_msk.repeat(B, 1, 1, 1)
        Bs_s_msk = Bs_s_msk.repeat(B, 1, 1, 1)

        # reshape to match scan helper (keep notebook’s magic constants)
        s_msk = s_msk.view(B, 4, 1024, L).permute(3, 0, 1, 2)
        dts_s_msk = dts_s_msk.view(B, 4, 1024, L).permute(3, 0, 1, 2).contiguous()
        Bs_s_msk = Bs_s_msk.view(B, 4, 16, L).permute(3, 0, 1, 2).contiguous()

        if force_fp32:
            s_msk = s_msk.float()
            dts_s_msk = dts_s_msk.float()
            Bs_s_msk = Bs_s_msk.float()

        if seq:
            out_y = []
            for i in range(4):
                yi = selective_scan_multi_input(
                    xs.view(B, K, -1, L)[:, i],
                    dts.view(B, K, -1, L)[:, i],
                    As.view(K, -1, N)[i],
                    Bs[:, i].unsqueeze(1),
                    Cs[:, i].unsqueeze(1),
                    Ds.view(K, -1)[i],
                    delta_bias=dt_projs_bias.view(K, -1)[i],
                    delta_softplus=True,
                ).view(B, -1, L)
                out_y.append(yi)
            out_y = torch.stack(out_y, dim=1)
        else:
            out_y = selective_scan_multi_input(
                xs,
                dts,
                As,
                Bs,
                Cs,
                Ds,
                u2s=s_msk,
                dt2s=dts_s_msk,
                B2s=Bs_s_msk,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
            ).view(B, K, -1, L)

        out_y_q = out_y[:, :, 1::2, :]
        inv_y = torch.flip(out_y_q[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y_q[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y_q[:, 0] + inv_y[:, 0] + wh_y + invwh_y

        y = y.transpose(dim0=1, dim1=2).contiguous()
        y = self.out_norm(y).view(B, H, W, -1)
        out = self.dropout(self.out_proj(y))
        return out, s_permuted


class IMISS(nn.Module, SS2Dv0):
    def __init__(
        self,
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        d_conv=3,
        conv_bias=True,
        dropout=0.0,
        bias=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        forward_type="v0",
        channel_first=False,
        **kwargs,
    ):
        nn.Module.__init__(self)
        kwargs.update(
            d_model=d_model,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            act_layer=act_layer,
            d_conv=d_conv,
            conv_bias=conv_bias,
            dropout=dropout,
            bias=bias,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_init_floor=dt_init_floor,
            initialize=initialize,
            forward_type=forward_type,
            channel_first=channel_first,
        )
        if forward_type in ["v0", "v0seq"]:
            self.__initv0__(seq=("seq" in forward_type), **kwargs)
        else:
            raise NotImplementedError(f"forward_type={forward_type} not supported in this refactor")


class MIVSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        channel_first: bool = False,
        ssm_d_state: int = 16,
        ssm_ratio: float = 2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias: bool = True,
        ssm_drop_rate: float = 0.0,
        ssm_init: str = "v0",
        forward_type: str = "v0",
        mlp_ratio: float = 4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        gmlp: bool = False,
        use_checkpoint: bool = False,
        post_norm: bool = False,
        _SS2D: type = IMISS,
        **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = _SS2D(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                dropout=ssm_drop_rate,
                initialize=ssm_init,
                forward_type=forward_type,
                channel_first=channel_first,
            )

        self.drop_path = DropPath(drop_path)

        if self.mlp_branch:
            _MLP = Mlp if not gmlp else gMlp
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = _MLP(
                in_features=hidden_dim,
                hidden_features=mlp_hidden_dim,
                act_layer=mlp_act_layer,
                drop=mlp_drop_rate,
                channels_first=channel_first,
            )

    def _forward(self, q: torch.Tensor, s: torch.Tensor, s_msk: torch.Tensor):
        if self.ssm_branch:
            if self.post_norm:
                q = q + self.drop_path(self.norm(self.op(q, s, s_msk)[0]))
            else:
                q = q + self.drop_path(self.op(self.norm(q), s, s_msk)[0])
        if self.mlp_branch:
            if self.post_norm:
                q = q + self.drop_path(self.norm2(self.mlp(q)))
            else:
                q = q + self.drop_path(self.mlp(self.norm2(q)))
        return q, self.op(q, s, s_msk)[1]

    def forward(self, q: torch.Tensor, s: torch.Tensor, s_msk: torch.Tensor):
        q = q.permute(0, 2, 3, 1)
        s = s.permute(0, 2, 3, 1)
        s_msk = s_msk.permute(0, 2, 3, 1).float()
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, q, s, s_msk)
        return self._forward(q, s, s_msk)
