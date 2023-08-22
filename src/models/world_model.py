from dataclasses import dataclass
from typing import Any, Optional, Tuple

from einops import rearrange
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import Batch
from .kv_caching import KeysValues
from .slicer import Embedder, Head
from .tokenizer import Tokenizer
from .transformer import Transformer, TransformerConfig
from utils import init_weights, LossWithIntermediateLosses


@dataclass
class WorldModelOutput:
    output_sequence: torch.FloatTensor
    logits_observations: torch.FloatTensor



class WorldModel(nn.Module):
    def __init__(self, obs_vocab_size: int, config: TransformerConfig) -> None:
        super().__init__()
        self.obs_vocab_size = obs_vocab_size
        config = TransformerConfig(tokens_per_block=768, max_blocks=3, attention="causal", num_layers=6, num_heads=8, embed_dim=256, embed_pdrop=0.1, resid_pdrop=0.1, attn_pdrop=0.1)
        self.transformer = Transformer(config)
        batch_size= 1
        
        
        sequence_length = 1536

# Create a tensor with ones for the known tokens and zeros for the unknown tokens
        known_steps = 768 
        block_mask = torch.cat([torch.ones(known_steps), torch.zeros(sequence_length)]).to(device='cuda:1')
        #print(block_mask.size())
        self.block_mask_tensor = block_mask.unsqueeze(0).expand(batch_size, -1).to(device='cuda:1')

        posit_emb_in=nn.Embedding(768, 256)
        posit_emb_out=nn.Embedding(1536, 256)
        self.image_embed=nn.Embedding(1024,256)

        self.Positional = posit_emb_in(torch.arange(768)).to(device='cuda:1')
        self.Predicted = posit_emb_out (torch.arange(1536)).to(device='cuda:1')
        #print("positional", self.Positional.size())
        self.pos_emb_in = self.Positional.unsqueeze(0).expand(batch_size, -1, -1).to(device='cuda:1')
        #print(self.pos_emb_in.size())
        self.pos_emb_out = self.Predicted.unsqueeze(0).expand(batch_size, -1, -1).to(device='cuda:1')
        #print("positional", self.pos_emb_out.size())


        self.head_observations = nn.Linear(256, 256)
           

            

            
        self.apply(init_weights)

        

    def forward(self, obs_tokens: torch.LongTensor) -> torch.FloatTensor:
        
        #assert num_steps <= self.config.max_tokens
        context_image = self.image_embed(self.obs_tokens[:, :768]).to(device='cuda:1')
        predicted_image= self.image_embed(self.obs_tokens[:, 768:]).to(device='cuda:1')
        combined_context_image = context_image + self.Positional
        combined_context_image.to(device='cuda:1')
        self.combined_predicted_image = predicted_image + self.Predicted
        self.combined_predicted_image.to(device='cuda:1')

        self.combined_image=torch.cat([combined_context_image, self.combined_predicted_image], dim=1)
        print("Combined Image", self.combined_image.size())
         # Apply block mask multiplication
        combining_image = self.combined_image * self.block_mask_tensor.unsqueeze(2)
        #print("COMBINED",combining_image.size())
        
        

        x = self.transformer(combining_image)
        logits_observations = self.head_observations(x)
        #print("x world", x.size())
        #print("logit", logits_observations.size())

        return x, logits_observations
    
    def compute_loss(self, batch: Batch, tokenizer: Tokenizer, **kwargs: Any) -> LossWithIntermediateLosses:
        with torch.no_grad():
            batch_obs = batch.unsqueeze(2)

            shape = batch_obs.shape
            # print(shape)
            self.obs_tokens= tokenizer.encode(batch_obs[:,:,:,:,:], should_preprocess=True).tokens 
            self.obs_tokens=self.obs_tokens.view(4, -1)
            # print(self.obs_tokens)

            x, logits_observations = self.forward(self.obs_tokens)

            target_obs = self.combined_predicted_image
            #print("Target observation", target_obs.size())
        
        # Compute the loss between the predicted observations and the real target observations
            predicted_obs = x
            # print("Predicted Obs", predicted_obs.size())
            loss_obs = F.cross_entropy((logits_observations.view(-1, logits_observations.size(-1))), (self.combined_image.view(-1,self.combined_image.size(-1))))
            print("Losses", loss_obs)

        return LossWithIntermediateLosses(loss_obs=loss_obs)
