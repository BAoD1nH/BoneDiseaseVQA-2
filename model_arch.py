import torch
import torch.nn as nn
from transformers import SwinModel, AutoModel, AutoConfig

class BoneDiseaseVQA(nn.Module):
    """
    Semi-open framework for Medical VQA.
    Combines Swin Transformer (vision) + BERT (text) + CMAN fusion + answer-querying decoder.
    """
    def __init__(
        self,
        vision_model_name: str,
        text_model_name: str,
        answer_classes: list,
        embed_dim: int = 1024,
        fusion_layers: int = 2,
        decoder_layers: int = 2,
        num_heads: int = 8,
    ):
        super().__init__()
        # Vision encoder (Swin Transformer)
        self.vision = SwinModel.from_pretrained(vision_model_name)
        vision_hidden = self.vision.config.hidden_size
        self.vision_proj = nn.Linear(vision_hidden, embed_dim)

        # Text encoder (BERT / ViHealthBERT)
        self.text = AutoModel.from_pretrained(text_model_name)
        text_hidden = self.text.config.hidden_size
        self.text_proj = nn.Linear(text_hidden, embed_dim)

        # Cross-modality fusion: TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4
        )
         
        self.fusion = nn.TransformerEncoder(encoder_layer, num_layers=fusion_layers)

        # Learnable answer embeddings (C x E)
        self.num_answers = len(answer_classes)
        self.answer_embeddings = nn.Parameter(torch.randn(self.num_answers, embed_dim))

        # Answer querying decoder: TransformerDecoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)

        # Classification head: maps each refined answer embedding to a logit
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, pixel_values, input_ids, attention_mask):
        # pixel_values: (B, C, H, W)
        # input_ids, attention_mask: (B, T)
        B = pixel_values.size(0)
        # 1) Vision features
        vision_out = self.vision(pixel_values=pixel_values)
        # last_hidden_state: (B, N, H_v)
        img_feats = vision_out.last_hidden_state  # sequence of region tokens
        img_feats = self.vision_proj(img_feats)  # (B, N, E)

        # 2) Text features
        txt_out = self.text(input_ids=input_ids, attention_mask=attention_mask)
        txt_feats = txt_out.last_hidden_state  # (B, T, H_t)
        txt_feats = self.text_proj(txt_feats)   # (B, T, E)

        # 3) Fusion: concat and TransformerEncoder
        fused = torch.cat([img_feats, txt_feats], dim=1)  # (B, N+T, E)
        # Transformer expects (S, B, E)
        fused = fused.permute(1, 0, 2)
        fused = self.fusion(fused)  # (S, B, E)
        fused = fused.permute(1, 0, 2)  # (B, S, E)

        # 4) Answer-querying decoder
        # prepare answer embeddings as tgt: (C, B, E)
        ans_emb = self.answer_embeddings.unsqueeze(1).expand(-1, B, -1)
        # memory = fused permuted to (S, B, E)
        memory = fused.permute(1, 0, 2)
        dec_out = self.decoder(tgt=ans_emb, memory=memory)  # (C, B, E)
        dec_out = dec_out.permute(1, 0, 2)  # (B, C, E)

        # 5) Classification logits
        logits = self.classifier(dec_out).squeeze(-1)  # (B, C)
        # probabilities = torch.sigmoid(logits)
        return logits