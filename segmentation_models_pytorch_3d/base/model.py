import torch
from . import initialization as init


class SegmentationModel(torch.nn.Module):
    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)

    def check_input_shape(self, x):

        h, w, d = x.shape[-3:]
        try:
            if self.encoder.strides is not None:
                hs, ws, ds = 1, 1, 1
                for stride in self.encoder.strides:
                    hs *= stride[0]
                    ws *= stride[1]
                    ds *= stride[2]
                if h % hs != 0 or w % ws != 0 or d % ds != 0:
                    new_h = (h // hs + 1) * hs if h % hs != 0 else h
                    new_w = (w // ws + 1) * ws if w % ws != 0 else w
                    new_d = (d // ds + 1) * ds if d % ds != 0 else d
                    raise RuntimeError(
                        f"Wrong input shape height={h}, width={w}, depth={d}. Expected image height and width and depth "
                        f"divisible by {hs}, {ws}, {ds}. Consider pad your images to shape ({new_h}, {new_w}, {new_d})."
                    )
            else:
                output_stride = self.encoder.output_stride
                if h % output_stride != 0 or w % output_stride != 0 or d % output_stride != 0:
                    new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
                    new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
                    new_d = (d // output_stride + 1) * output_stride if d % output_stride != 0 else d
                    raise RuntimeError(
                        f"Wrong input shape height={h}, width={w}, depth={d}. Expected image height and width and depth "
                        f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w}, {new_d})."
                    )
        except:
            pass

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x)

        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        out = list()
        
        for i, item in enumerate(decoder_output):
            mask = self.segmentation_heads[i](item)
            out.append(mask)

        return out