import torch 
import torch.nn as nn 
from transformers import CLIPProcessor, CLIPModel
from model.Qformer import BertForMaskedLM
from transformers.models.bert.configuration_bert import BertConfig
class QFormer(nn.Module):
    def __init__(self, 
                 visual_dim=768,
                 text_dim=50,
                 num_queries=32,
                 num_heads=8,
                 num_layers=3):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # query_embeds 
        self.query_embeds = nn.Parameter(torch.randn(num_queries, visual_dim))
        nn.init.xavier_uniform_(self.query_embeds)

        # self-attention layers and cross-attention layers
        self.self_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(visual_dim, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(visual_dim, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        for layer in self.self_attn_layers + self.cross_attn_layers:
            nn.init.xavier_uniform_(layer.in_proj_weight)
            nn.init.constant_(layer.in_proj_bias, 0)

        # linear layers
        self.linear_layers = nn.ModuleList([
            nn.Linear(visual_dim, visual_dim)
            for _ in range(num_layers)
        ])
        for linear in self.linear_layers:
            nn.init.kaiming_normal_(linear.weight, mode='fan_out', nonlinearity='relu')

        # text projection layer
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, visual_dim // 4),
            nn.LayerNorm(visual_dim // 4),
            nn.GELU(),
            nn.Linear(visual_dim // 4, visual_dim),
            nn.LayerNorm(visual_dim),
            nn.Dropout(0.3)
        ).to(self.device)
        for layer in [0, 3]:  # 更新初始化层索引
            nn.init.kaiming_normal_(self.text_proj[layer].weight, 
                                  mode='fan_in', 
                                  nonlinearity='relu')
            if self.text_proj[layer].bias is not None:
                nn.init.constant_(self.text_proj[layer].bias, 0.01)
        
        # text self-attention layer
        self.text_self_attn = nn.MultiheadAttention(visual_dim, num_heads, batch_first=True)
        nn.init.xavier_uniform_(self.text_self_attn.in_proj_weight)
        nn.init.constant_(self.text_self_attn.in_proj_bias, 0)

        # visual projection layer
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, visual_dim//2),
            nn.LayerNorm(visual_dim//2),
            nn.GELU(),
            nn.Linear(visual_dim//2, visual_dim//4),
            nn.LayerNorm(visual_dim//4),
            nn.GELU(),
            nn.Linear(visual_dim//4, visual_dim)
        ).to(self.device)
        for layer in [0, 3, 6]:  
            nn.init.kaiming_normal_(self.visual_proj[layer].weight, 
                                  mode='fan_out', 
                                  nonlinearity='relu')
            if self.visual_proj[layer].bias is not None:
                nn.init.constant_(self.visual_proj[layer].bias, 0.01)

        # gate layer
        self.gate = nn.Sequential(
            nn.Linear(visual_dim*3, 1),  
            nn.Sigmoid()
        ).to(self.device)
        nn.init.xavier_normal_(self.gate[0].weight)
        nn.init.constant_(self.gate[0].bias, 0.0)

        # self_attn_layer2
        self.self_attn = nn.ModuleList([
            nn.MultiheadAttention(visual_dim, num_heads, batch_first=True)
            for _ in range(3)
        ])

        # normalization layer
        self.norm = nn.LayerNorm(visual_dim)
        # dropout layer
        self.dropout = nn.Dropout(0.4)
        self.text_dropout = nn.Dropout(0.2)
        self.attn_dropout = nn.Dropout(0.4)

    def forward(self, visual_feats, text_feats):
        B = visual_feats.size(0)
        queries = self.query_embeds.unsqueeze(0).expand(B, -1, -1)
        text_feats = self.norm(self.text_proj(text_feats))
        visual_feats = self.norm(visual_feats)
        text_feats = self.text_dropout(text_feats)
        visual_feats = self.visual_proj(visual_feats)
        if  text_feats is None:
            for i in range(len(self.cross_attn_layers)):
                combined = queries
                self_attn_output, _ = self.self_attn_layers[i](combined, combined, combined)
                self_attn_queries = self_attn_output[:, :queries.size(1), :]
                cross_attn_output, _ = self.cross_attn_layers[i](
                    self_attn_queries, 
                    self.attn_dropout(visual_feats),
                    self.attn_dropout(visual_feats)
                )
                residual = queries
                queries = self.norm(cross_attn_output + residual)
                queries = self.linear_layers[i](queries) 
                queries = self.dropout(queries)
                queries = self.norm(queries * 0.8 + queries.detach() * 0.2)
            visual_ctx = visual_feats.mean(1, keepdim=True)
            fused_feats = visual_ctx 
            for i in range(3):
                residual = fused_feats
                fused_feats, _ = self.self_attn[i](fused_feats, fused_feats, fused_feats)  # 添加元组解包
                fused_feats = fused_feats + residual
                fused_feats = self.norm(fused_feats)
            return fused_feats



        for i in range(len(self.cross_attn_layers)):
            combined = torch.cat([queries, text_feats], dim=1)
            self_attn_output, _ = self.self_attn_layers[i](combined, combined, combined)
            self_attn_queries = self_attn_output[:, :queries.size(1), :]
            cross_attn_output, _ = self.cross_attn_layers[i](
                self_attn_queries, 
                self.attn_dropout(visual_feats),
                self.attn_dropout(visual_feats)
            )
            residual = queries
            queries = self.norm(cross_attn_output + residual)
            queries = self.linear_layers[i](queries) 
            queries = self.dropout(queries)
            queries = self.norm(queries * 0.8 + queries.detach() * 0.2)
        visual_ctx = visual_feats.mean(1, keepdim=True)
        text_ctx = text_feats.mean(1, keepdim=True)
        gate_input = torch.cat([
            visual_ctx.expand(-1, queries.size(1), -1), 
            text_ctx.expand(-1, queries.size(1), -1),
            text_feats.mean(1, keepdim=True).expand(-1, queries.size(1), -1)
        ], dim=-1)
        
        gate = self.gate(gate_input) 
        fused_feats = gate * (queries + text_ctx) + (1 - gate) * visual_ctx
        for i in range(3):
            residual = fused_feats
            fused_feats, _ = self.self_attn[i](fused_feats, fused_feats, fused_feats)  
            fused_feats = fused_feats + residual
            fused_feats = self.norm(fused_feats)
            

        return fused_feats



class EmoEvent(nn.Module):
    def __init__(self, num_classes=6, image_size=224, frames = 5):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.frames = 5
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # classifier
        self.proj = nn.Sequential(
            nn.Dropout(0.1),
            nn.LayerNorm(15),
            nn.Linear(15, 12),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(12, 7),
            nn.LayerNorm(7),
            nn.Linear(7, num_classes, bias=False) 
        ).to(self.device)
        nn.init.kaiming_normal_(self.proj[2].weight, mode='fan_out', nonlinearity='relu') 
        nn.init.kaiming_normal_(self.proj[5].weight, mode='fan_out', nonlinearity='relu')  
        nn.init.normal_(self.proj[-1].weight, mean=0.0, std=0.02) 

        # attention pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(768, 256),
            nn.Tanh(),
            nn.Linear(256, 1, bias=False),
            nn.Softmax(dim=1)
        ).to(self.device)
        nn.init.xavier_normal_(self.attention_pool[0].weight)
        nn.init.constant_(self.attention_pool[0].bias, 0.0)
        nn.init.xavier_normal_(self.attention_pool[2].weight)


    def forward(self, llm_embeds=None, vit_embeds=None, labels=None):
        vit_embeds = vit_embeds.to(self.device)
        llm_embeds = llm_embeds
        inputs = self.processor(text=llm_embeds, images=vit_embeds, return_tensors="pt", padding=True).to(self.device)
        outputs = self.clip_model(**inputs)
        x = outputs.logits_per_image
        # print(x.shape)

        

        logits = self.proj(x)
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.proj[-1].out_features), 
                          labels.view(-1))
            return loss  
        else:
            return logits
        
