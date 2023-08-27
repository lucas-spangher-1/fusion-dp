from .s4_hazy import S4Block
from omegaconf import OmegaConf


class S4(S4Block):
    """Wrapper for S4Block"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        data_dim: int,
        kernel_cfg: OmegaConf,
        conv_cfg: OmegaConf,
        mask_cfg: OmegaConf,
    ):
        assert (
            in_channels == out_channels
        ), f"S4 only supports the same # of in, out channels, given {in_channels}, {out_channels}"
        assert data_dim == 1, f"S4 only supports 1-d data. Given {data_dim}"
        s4args = getattr(conv_cfg, "s4", {})
        super().__init__(d_model=in_channels, **s4args)

    def forward(self, x):
        res, _ = super().forward(x)
        return res
