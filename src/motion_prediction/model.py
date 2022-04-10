import torch.nn as nn
import torch

from src.detector.utils import normalize_boxes, denormalize_boxes
from src.motion_prediction.kalman import FullBoxFilter


class KalmanPredictor(nn.Module):
    def __init__(self, process_variance, measurement_variance, **kwargs):
        super().__init__()
        self.kalman = FullBoxFilter(
            process_variance=process_variance,
            measurement_variance=measurement_variance,
        )

    @classmethod
    def from_config(cls, hyperparams):
        self = cls.__new__(cls)
        self.__init__(**hyperparams)
        return self

    def forward(self, trajs, future_lens):
        """
        trajs  
            - torch.Tensor [bs, history_len, 4]
            - or list of torch.Tensor [history_len, 4] with different lengths

        preds
            - torch.Tensor [bs, future_len, 4]
        """
        preds = []
        for traj, future_len in zip(trajs, future_lens):
            pred = self.kalman.predict(traj, future_len=future_len)
            self.kalman.reset_state()
            preds.append(pred)
        return preds


class MotionPredictor(nn.Module):
    def __init__(
        self, input_dim=4, hidden_dim=32, output_dim=4, future_len=1, **kwargs
    ):
        super().__init__()
        self.future_len = future_len
        self.net = EncoderDecoder(input_dim, hidden_dim, output_dim)

    @classmethod
    def from_config(cls, hyperparams):
        self = cls.__new__(cls)
        self.__init__(**hyperparams)
        return self

    def forward(self, hist, future_len=None):
        if future_len is None:
            future_len = self.future_len

        anchor = hist[:, -1]
        hist_norm = normalize_boxes(anchor_boxes=anchor, boxes=hist)
        fut_norm = self.net(hist_norm, future_len=future_len)
        fut = denormalize_boxes(anchor_boxes=anchor, normalized_boxes=fut_norm)
        return fut


class EncoderDecoder(nn.Module):
    # TODO: add activations / dropout / batchnorm etc.
    # TODO: add VAE so that tracker can play through different scenarios

    def __init__(self, input_dim=4, hidden_dim=32, output_dim=4, fl=1):
        super().__init__()
        self.input_embedder = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim), nn.ReLU()
        )
        self.gru_encoder = nn.GRU(
            input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True
        )
        self.gru_decoder = nn.GRU(
            input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True
        )
        self.output_embedder = nn.Linear(
            in_features=hidden_dim, out_features=output_dim
        )

    def forward(self, x, future_len):
        """
        bs : batch size
        fl : future length (timesteps, not seconds)
        hl: history length (timesteps, not seconds)


        Arguments
        ---------
        x 
            - shape: [bs, min_hl, 4]
            - 4 features: cx, cy, w, h

        Returns
        -------
        y
            - shape: [bs, ph, 4]
            - 4 features: cx, cy, w, h
        """

        x = self.input_embedder(x)
        out, _ = self.gru_encoder(x)
        latent = out[:, -1]

        dec_out_seq = []
        for step in range(future_len):
            if step == 0:
                inp = latent.unsqueeze(1)
            else:
                inp = torch.cat([inp, last_out], dim=1)
            out, h = self.gru_decoder(inp, latent.unsqueeze(0))
            last_out = out[:, [-1]]
            dec_out_seq.append(last_out)
        dec_out_seq = torch.cat(dec_out_seq, dim=1)
        out_seq = self.output_embedder(dec_out_seq)
        return out_seq
