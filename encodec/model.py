from typing import List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential

from einops import rearrange, reduce
from einops.layers.torch import Rearrange


# TODO: Finish documentation for undocumented modules.


def fast_mse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Fast mean squared error."""

    return torch.sum(x ** 2, dim=1, keepdim=True) \
         + torch.sum(y ** 2, dim=1) - 2*(x @ y.T)


class Codebook(nn.Module):
    """Codebook.

    Implements a codebook as described in (Oord et al., 2017). The codebook
    holds a set of vectors that are optimized to minimize the expected squared
    distance to vectors seen during training.

    Example
    -------
    >>> codebook = Codebook(codebook_size=64, codebook_channels=3)
    >>> x = torch.randn((1, 3, 256, 256))
    >>> x, indices, codebook_loss, commitment_loss = codebook(x)
    """

    def __init__(
        self,
        codebook_size: int,
        codebook_channels: int,
    ) -> None:
        """Initialize the module.

        Parameters
        ----------
        codebook_size : int
            The codeboook size.
        codebook_channels : int
            The codebook channels (i.e., channels for each codebook vector).
        """

        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=codebook_size,
            embedding_dim=codebook_channels,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            The unquantized tensor.

        Returns
        -------
        z : torch.Tensor
            The quantized tensor.
        sequence : torch.Tensor
            The sequence of codebook indices.
        codebook_loss : torch.Tensor
            The codebook loss. MSE w.r.t. the codebook parameters.
        commitment_loss : torch.Tensor
            The commitment loss. MSE w.r.t. the encoder parameters.
        """

        B, C, H, W = x.shape

        x = rearrange(x, 'b c h w -> (b h w) c')
        indices = fast_mse(x, self.embedding.weight).argmin(dim=1).detach()
        z = self.embedding(indices)

        codebook_loss = F.mse_loss(z, x.detach())
        commitment_loss = F.mse_loss(x, z.detach())

        z = x + (z - x).detach()  # STE.
        z = rearrange(z, '(b h w) c -> b c h w', h=H, w=W)
        sequence = indices.view(B, H, W)

        return z, sequence, codebook_loss, commitment_loss


class RVQ(nn.Module):
    """RFSQ.

    Implements Residual Vector Quantization (RVQ) (Juang et al., 1982). RVQ is a
    vector quantization scheme that uses multiple codebooks to recursively
    compress residuals.

    Exmple
    ------
    >>> module = RVQ(
    ...     codebook_channels=3,
    ...     codebook_sizes=(64, 32, 16, 8),  # 4 codebooks of varying sizes.
    ... )
    >>> x = torch.randn((1, 3, 256, 256))
    >>> x = module(x)
    """

    def __init__(
        self,
        codebook_sizes: Tuple[int, ...],
        codebook_channels: int,
    ) -> None:
        """Initializes the module.

        Parameters
        ----------
        codebook_sizes : Tuple[int, ...]
            The size of each codebook.
        codebook_chanels : int
            The number of codebook channels.
        """

        super().__init__()

        self.codebooks = nn.ModuleList([
            Codebook(
                codebook_size=codebook_size,
                codebook_channels=codebook_channels,
            )
            for codebook_size in codebook_sizes
        ])

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
    ]:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            The unquantized tensor.

        Returns
        -------
        outputs : List[torch.Tensor]
            The outputs from each codebook.
        residuals : List[torch.Tensor]
            The residuals from each codebook.
        sequences : List[torch.Tensor]
            The sequences from each codebook.
        codebook_losses : List[torch.Tensor]
            The codebook losses from each codebook.
        commitment_losses : List[torch.Tensor]
            The commitment losses from each codebook.
        """

        outputs = []
        residuals = []
        sequences = []
        codebook_losses = []
        commitment_losses = []

        for codebook in self.codebooks:
            z, sequence, codebook_loss, commitment_loss = codebook(x)
            residual = torch.log(x - z)

            outputs.append(z)
            residuals.append(residual)
            sequences.append(sequence)
            codebook_losses.append(codebook_loss)
            commitment_losses.append(commitment_loss)

            x = residual

        return (
            outputs,
            residuals,
            sequences,
            codebook_losses,
            commitment_losses,
        )


# Macros.

def Convolution1(input_channels: int, output_channels: int) -> nn.Module:

    return nn.Conv1d(
        in_channels=input_channels,
        out_channels=output_channels,
        kernel_size=1,
        stride=1,
        padding=0,
    )


def Convolution3(input_channels: int, output_channels: int) -> nn.Module:

    return nn.Conv1d(
        in_channels=input_channels,
        out_channels=output_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )


def Convolution4(input_channels: int, output_channels: int) -> nn.Module:

    return nn.Conv1d(
        in_channels=input_channels,
        out_channels=output_channels,
        kernel_size=4,
        stride=2,
        padding=1,
        bias=False,
    )


def Normalization(channels: int) -> nn.Module:

    return nn.Sequential(
        Rearrange('b e t -> b t e'),
        nn.LayerNorm(
            normalized_shape=channels,
        ),
        Rearrange('b t e -> b e t'),
    )


def Repeat(module, channels_list: List[int]) -> nn.Module:

    return nn.Sequential(*(
        module(
            input_channels=input_channels,
            output_channels=output_channels,
        ) for input_channels, output_channels in zip(
            channels_list[: -1],
            channels_list[1 :],
        )
    ))


# Modules.

class ResidualBlock(nn.Module):

    def __init__(self, channels: int) -> None:
        super().__init__()

        self.normalization = Normalization(channels=channels)

        self.convolution = Convolution3(
            input_channels=channels,
            output_channels=channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        z = self.normalization(x)
        z = F.leaky_relu(z)
        z = self.convolution(z)

        return x + z


class ResNetBlock(nn.Module):

    def __init__(self, channels: int) -> None:
        super().__init__()

        self.residual_block_1 = ResidualBlock(channels=channels)
        self.residual_block_2 = ResidualBlock(channels=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.residual_block_1(x)
        x = self.residual_block_2(x)

        return x


class UpsampleBlock(nn.Module):

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()

        self.normalization = Normalization(channels=input_channels)

        self.convolution = Convolution3(
            input_channels=input_channels,
            output_channels=output_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.normalization(x)
        x = F.leaky_relu(x)
        # x = self.upsample(x)
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        x = self.convolution(x)

        return x


class DownsampleBlock(nn.Module):

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()

        self.normalization = Normalization(channels=input_channels)

        self.convolution = Convolution4(
            input_channels=input_channels,
            output_channels=output_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.normalization(x)
        x = F.leaky_relu(x)
        x = self.convolution(x)

        return x


class UpBlock(nn.Module):

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()

        self.resnet_block = ResNetBlock(channels=input_channels)

        self.upsample_block = UpsampleBlock(
            input_channels=input_channels,
            output_channels=output_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.resnet_block(x)
        x = self.upsample_block(x)

        return x


class DownBlock(nn.Module):

    def __init__(self, input_channels: int, output_channels: int) ->  None:
        super().__init__()

        self.resnet_block = ResNetBlock(channels=input_channels)

        self.downsample_block = DownsampleBlock(
            input_channels=input_channels,
            output_channels=output_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.resnet_block(x)
        x = self.downsample_block(x)

        return x


class MiddleBlock(nn.Module):

    def __init__(self, channels: int) -> None:
        super().__init__()

        self.resnet_block_1 = ResNetBlock(channels=channels)
        self.resnet_block_2 = ResNetBlock(channels=channels)
        # self.attention_block = AttentionBlock(channels=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.resnet_block_1(x)
        # x = self.attention_block(x)
        x = self.resnet_block_2(x)

        return x


class Encoder(nn.Module):

    def __init__(self, channels_list: List[int]) -> None:
        super().__init__()

        self.down_blocks = Repeat(module=DownBlock, channels_list=channels_list)
        self.middle_block = MiddleBlock(channels=channels_list[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.down_blocks(x)
        x = self.middle_block(x)

        return x


class Decoder(nn.Module):

    def __init__(self, channels_list: List[int]) -> None:
        super().__init__()

        self.up_blocks = Repeat(module=UpBlock, channels_list=channels_list)
        self.middle_block = MiddleBlock(channels=channels_list[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.middle_block(x)
        x = self.up_blocks(x)

        return x


@dataclass(frozen=True)
class VQVAEConfiguration:
    input_channels: int
    output_channels: int
    latent_channels: int
    encoder_channels_list: List[int]
    decoder_channels_list: List[int]
    codebook_sizes: Tuple[int, ...]


class VQVAE(nn.Module):

    def __init__(self, configuration: VQVAEConfiguration) -> None:
        super().__init__()

        self.encoder = Encoder(channels_list=configuration.encoder_channels_list)
        self.decoder = Decoder(channels_list=configuration.decoder_channels_list)

        # Input to encoder.

        self.convolution_1 = Convolution3(
            input_channels=configuration.input_channels,
            output_channels=configuration.encoder_channels_list[0],
        )

        # Encoder to latent.

        self.convolution_2 = Convolution3(
            input_channels=configuration.encoder_channels_list[-1],
            output_channels=configuration.latent_channels,
        )

        # Latent to decoder.

        self.convolution_3 = Convolution3(
            input_channels=configuration.latent_channels,
            output_channels=configuration.decoder_channels_list[0],
        )

        # Decoder to output.

        self.convolution_4 = Convolution3(
            input_channels=configuration.decoder_channels_list[-1],
            output_channels=configuration.output_channels,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:

        x = self.convolution_1(x)
        x = self.encoder(x)
        x = self.convolution_2(x)
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:

        x = self.convolution_3(z)
        x = self.decoder(x)
        x = self.convolution_4(x)

        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        z = self.encode(x)
        x = self.decode(z)

        return x, z
