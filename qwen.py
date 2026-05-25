import torch
from torch import nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(
            self, 
            dim: int, 
            eps: int=1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def _norm(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps)
    def forward(self, x):
        output = self._norm(x)
        output = output * self.weight
        return output
        


class RoPE(nn.Module):
    def __init__(
            self,
            config,
    ):
        super().__init__()
        self.dim = config.head_dim
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, self.dim, 2)/ self.dim))
        self.pos_rotary_factor = 1.0
        self.theta = config.theta*self.pos_rotary_factor
        self.mrope_section = list(getattr(config, "mrope_section", [self.dim]))

    def apply_interleaved_mrope(self, freq: torch.Tensor) -> torch.Tensor:
        """Interleave the 3-way MRoPE streams into a single rotary embedding tensor."""
        if freq.ndim != 4 or freq.shape[0] != 3:
            raise ValueError("freq must have shape [3, batch, seq_len, head_dim/2]")

        if sum(self.mrope_section) != self.dim:
            raise ValueError(
                f"Invalid mrope_section {self.mrope_section}: expected sum to equal head_dim={self.dim}"
            )

        # Duplicate frequencies to build the full rotary dimension before section interleaving.
        full_freq = torch.cat((freq, freq), dim=-1)
        sections = full_freq.split(self.mrope_section, dim=-1)
        return torch.cat([chunk[i % 3] for i, chunk in enumerate(sections)], dim=-1)

    
    
    def forward (self, x: torch.Tensor, pos_ids: torch.Tensor):
        """Apply RoPE to the input tensor x using the provided position ids.
        Args:            
            x: A tensor of shape (batch_size,number of heads, seq_len, dim, head size) representing the input embeddings
            pos_ids: A tensor of shape (seq_len) containing the position ids for each token
            Returns:
            A tensor of the same shape as x, with RoPE applied to the input embeddings.
            """
        
        # Ensure pos_ids is [batch_size, seq_len]
        if pos_ids.dim() == 1:
            pos = pos_ids.unsqueeze(0).expand(x.shape[0], -1)
        else:
            pos = pos_ids

        # inv_freq: [dim/2]
        inv = self.inv_freq.to(x.device).to(x.dtype)

        # angles: [batch, seq_len, dim/2]
        angles = pos.float().unsqueeze(-1) * inv.unsqueeze(0).unsqueeze(0)

        # duplicate to full rotary dim: [batch, seq_len, dim]
        angles_full = torch.cat([angles, angles], dim=-1)

        return angles_full.cos().to(dtype=x.dtype), angles_full.sin().to(dtype=x.dtype)
            
def rotate_half(x: torch.Tensor):
    """Rotate the input tensor x by splitting it in half along the last dimension and swapping the halves with a sign change.
    Args:
        x: A tensor of shape (..., dim) where dim is the last dimension to be rotated.
    Returns:
        A tensor of the same shape as x, with the last dimension rotated by half.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply RoPE to the query and key tensors using the provided cosine and sine embeddings.
    Args:
        q: A tensor of shape (batch_size, num_heads, seq_len, head_dim) representing the query embeddings
        k: A tensor of shape (batch_size, num_heads, seq_len, head_dim) representing the key embeddings
        cos: A tensor of shape (batch_size, seq_len, head_dim) containing the cosine RoPE embeddings
        sin: A tensor of shape (batch_size, seq_len, head_dim) containing the sine RoPE embeddings
    Returns:
        A tuple of tensors (q_rot, k_rot) where q_rot and k_rot are the query and key tensors with RoPE applied, 
        each of shape (batch_size, num_heads, seq_len, head_dim).
    """
    # q,k: [batch_size, num_heads, seq_len, head_dim]
    # cos,sin: [batch_size, seq_len, head_dim]
    cos = cos.unsqueeze(1)  # [batch_size, 1, seq_len, head_dim]
    sin = sin.unsqueeze(1)  # [batch_size, 1, seq_len, head_dim]
    rotary_dim = cos.shape[-1]
    q1, q2 = q[..., :rotary_dim], q[..., rotary_dim:]
    k1, k2 = k[..., :rotary_dim], k[..., rotary_dim:]
    q1 =  q1 * cos - rotate_half(q1) * sin
    k1 =  k1 * cos - rotate_half(k1) * sin
    q_rot = torch.cat( (q1, q2), dim=-1)
    k_rot = torch.cat( (k1, k2), dim=-1)
    return q_rot, k_rot

ACT2FN = {
    "silu": F.silu,
    "swish": F.silu,
    "gelu": F.gelu,
    "relu": F.relu,
}


def torch_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    recurrent_state: torch.Tensor | None = None,
    cache: dict | None = None,
):
    """Minimal recurrent gated delta rule used by the local test harness.

    This keeps the tensor plumbing consistent with GatedDeltaNet without depending on
    an external kernel implementation.
    """
    del cache

    if q.shape[2] != v.shape[2]:
        if v.shape[2] % q.shape[2] == 0:
            repeat_factor = v.shape[2] // q.shape[2]
            q = q.repeat_interleave(repeat_factor, dim=2)
            k = k.repeat_interleave(repeat_factor, dim=2)
        elif q.shape[2] % v.shape[2] == 0:
            repeat_factor = q.shape[2] // v.shape[2]
            v = v.repeat_interleave(repeat_factor, dim=2)
            beta = beta.repeat_interleave(repeat_factor, dim=2)
            g = g.repeat_interleave(repeat_factor, dim=2)
        else:
            raise ValueError(
                f"Incompatible head counts for q/k ({q.shape[2]}) and v ({v.shape[2]})"
            )

    batch_size, seq_len, num_heads, head_dim = v.shape

    if recurrent_state is None:
        recurrent_state = torch.zeros(
            batch_size,
            num_heads,
            head_dim,
            device=v.device,
            dtype=v.dtype,
        )

    outputs = []
    for step in range(seq_len):
        q_step = q[:, step]
        k_step = k[:, step]
        v_step = v[:, step]
        g_step = torch.sigmoid(g[:, step]).unsqueeze(-1)
        beta_step = beta[:, step].unsqueeze(-1)

        delta = torch.tanh(q_step * k_step)
        recurrent_state = recurrent_state + beta_step * (v_step + delta)
        outputs.append(g_step * recurrent_state)

    attn_core = torch.stack(outputs, dim=1)
    return attn_core, recurrent_state


def build_causal_mask(
    attention_mask: torch.Tensor | None,
    batch_size: int,
    query_length: int,
    kv_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    min_value = torch.finfo(dtype).min
    causal = torch.full((query_length, kv_length), min_value, device=device, dtype=dtype)
    causal = torch.triu(causal, diagonal=1 + kv_length - query_length)

    causal = causal.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, query_length, kv_length)
    if attention_mask is None:
        return causal

    padding_mask = (1.0 - attention_mask[:, None, None, :].to(dtype)) * min_value # don't look to PAD tokens
    return causal + padding_mask 



class SelfAttention(nn.Module):
    def __init__(
            self,
            config,
            layer_idx,
    ):
        super().__init__()
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_attention_heads // self.num_kv_heads
        self.scaling= self.head_dim ** -0.5

        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)

        self.q_proj = nn.Linear(self.hidden_size, 
                                self.num_attention_heads * self.head_dim *2,
                                bias=config.attention_bias)
        self.k = nn.Linear(self.hidden_size, 
                                self.num_kv_heads * self.head_dim,
                                bias=config.attention_bias)
        self.v = nn.Linear(self.hidden_size, 
                                self.num_kv_heads * self.head_dim,
                                bias=config.attention_bias)
        self.proj_out = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

    def forward(self,
                hidden_states: torch.Tensor,
                pos_embeddings: tuple,
                attention_mask: torch.Tensor | None=None,
                cache: dict = None,
                ):
        """Compute self-attention for the input hidden states. 
        math explanation:
        q = W_q * hidden_states, k = W_k * hidden_states, v = W_v * hidden_states, 
        attn_weights = (q @ k.transpose(-2, -1)) / sqrt(head_dim) + pos_embeddings
        attn_weights = attn_weights.masked_fill(attention_mask == 0, -inf)
        attn_probs = softmax(attn_weights, dim=-1)
        attn_output = attn_probs @ v
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, num_attention_heads * head_dim)
        attn_output = proj_out(attn_output)

        """
        # hidden_states: [batch_size, seq_len, hidden_size]
        # pos_embeddings: tuple of (cos, sin) each of shape [batch_size, seq_len, head_dim]
        # attention_mask: [batch_size,num_attention_heads, qeuery_len, kv_len]
        batch_size, seq_len, _ = hidden_states.shape
        
        q_proj = self.q_proj(hidden_states) # [batch_size, seq_len, num_attention_heads*head_dim*2]
        q, gate =torch.chunk(q_proj, 2, dim=-1) # [batch_size, seq_len, num_attention_heads*head_dim] each
        gate = gate.reshape(batch_size, seq_len, -1) 
        
        
        q = self.q_norm(q.reshape(batch_size, seq_len, self.num_attention_heads, self.head_dim)).transpose(1, 2) # [batch_size, num_attention_heads, seq_len, head_dim]
        k = self.k_norm(self.k(hidden_states).reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)).transpose(1, 2) # [batch_size, num_kv_heads, seq_len, head_dim]
        v = self.v(hidden_states).reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2) # [batch_size, num_kv_heads, seq_len, head_dim]

        # Apply RoPE
        cos, sin = pos_embeddings
        q,k = apply_rope(q, k, cos, sin)

        if cache is not None:
            k,v = cache.update(k, v, self.layer_idx)

        k = k.repeat_interleave(self.num_kv_groups, dim=1) # [batch_size, num_attention_heads, seq_len, head_dim]
        v = v.repeat_interleave(self.num_kv_groups, dim=1) # [batch_size, num_attention_heads, seq_len, head_dim]
        
        # compute attention
        attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scaling # [batch_size, num_attention_heads, seq_len, kv_seq_len]
        
        if attention_mask is not None:
            mask = attention_mask
            # Normalize mask to boolean and expand to attn_weights dims
            if mask.dtype != torch.bool:
                mask = mask != 0
            while mask.dim() < attn_weights.dim():
                mask = mask.unsqueeze(1)
            if mask.shape != attn_weights.shape:
                mask = mask.expand_as(attn_weights)
            attn_weights = attn_weights.masked_fill(~mask, float("-inf"))
        
        attn_probs = F.softmax(attn_weights, dim=-1) # [batch_size, num_attention_heads, seq_len, kv_seq_len]
        attn_output = torch.matmul(attn_probs, v) # [batch_size, num_attention_heads, seq_len, head_dim]

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1) # [batch_size, seq_len, num_attention_heads*head_dim]

        attn_output = attn_output * torch.sigmoid(gate) # apply gating in hidden dimension
        attn_output = self.proj_out(attn_output) # [batch_size, seq_len, hidden_size]
        
        return attn_output



class GatedDeltaNet(nn.Module):
    def __init__(
            self,
            config,
            layer_idx,
    ):
        super().__init__()
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_k_heads = getattr(config, "num_k_heads", config.num_v_heads)
        self.num_v_heads = config.num_v_heads
        self.head_k_dim = config.head_k_dim
        self.head_v_dim = config.head_v_dim

        self.k_dim = self.num_k_heads * self.head_k_dim
        self.v_dim = self.num_v_heads * self.head_v_dim

        self.conv_dim= self.k_dim * 2 + self.v_dim
        self.kernel_size = getattr(config, "linear_conv_kernel_size", 1)

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.log(torch.empty(self.num_v_heads).uniform_(0, 16)))

        self.norm = RMSNorm(self.hidden_size, config.rms_norm_eps)
        self.qkv= nn.Linear(self.hidden_size, 
                            self.conv_dim,
                            bias=config.attention_bias)
        
        self.conv1d = nn.Conv1d(in_channels=self.conv_dim, 
                                out_channels=self.conv_dim, 
                                kernel_size=config.linear_conv_kernel_size, 
                                padding=config.linear_conv_kernel_size//2,
                                bias=config.attention_bias)
        
        self.proj_out = nn.Linear(self.v_dim, 
                                  self.hidden_size, 
                                  bias=config.attention_bias)
        
        self.z =nn.Linear(self.hidden_size, 
                          self.v_dim, 
                          bias=config.attention_bias)
        
        self.b = nn.Linear(self.hidden_size,
                            self.num_v_heads ,
                            bias=config.attention_bias)
        
        self.a = nn.Linear(self.hidden_size,
                            self.num_v_heads ,
                            bias=config.attention_bias)
        
    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor | None=None,
                cache: dict = None,
                ):
        """Compute Gated DeltaNet attention for the input hidden states. 
        math explanation:
        qkv = W_qkv * hidden_states
        q,k,v = split(qkv)
        conv_out = conv1d(q,k,v)
        z = W_z * hidden_states
        b = W_b * hidden_states
        a = W_a * hidden_states
        attn_output = proj_out(conv_out) * sigmoid(a) + z * sigmoid(b)

        """
        # hidden_states: [batch_size, seq_len, hidden_size]
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask[:,:,None]
        
        qkv = self.qkv(self.norm(hidden_states)).transpose(1, 2) # [batch_size, conv_dim, seq_len]
        
        conv_state = cache.conv_state[self.layer_idx] if cache is not None else None
        reccurrent_state = cache.recurrent_state[self.layer_idx] if cache is not None else None

        if cache is not None:
            conv_pad = max(self.kernel_size - qkv.shape[-2], 0)
            qkv = F.pad(qkv, (conv_pad, 0), value=0.0)  
            cache.conv_state[self.layer_idx] = conv_state

        qkv = F.silu(self.conv1d(qkv)[:, :, -hidden_states.shape[1]:]) # [batch_size, conv_dim, seq_len]
        qkv = qkv.transpose(1, 2) # [batch_size, seq_len, conv_dim]

        q,k,v = torch.split(qkv, [self.k_dim, self.k_dim, self.v_dim], dim=-1) # [batch_size, seq_len, k_dim] each

        q= q.reshape(hidden_states.shape[0], hidden_states.shape[1], self.num_k_heads, self.head_k_dim) # [batch_size, seq_len, num_k_heads, head_k_dim]
        k= k.reshape(hidden_states.shape[0], hidden_states.shape[1], self.num_k_heads, self.head_k_dim) # [batch_size, seq_len, num_k_heads, head_k_dim]
        v= v.reshape(hidden_states.shape[0], hidden_states.shape[1], self.num_v_heads, self.head_v_dim) # [batch_size, seq_len, num_v_heads, head_v_dim]

        z= self.z(hidden_states) # [batch_size, seq_len, v_dim]
        z= z.reshape(hidden_states.shape[0], hidden_states.shape[1],-1, self.head_v_dim) # [batch_size, seq_len, num_v_heads, head_v_dim]
        b= self.b(hidden_states) # [batch_size, seq_len, num_v_heads]
        a= self.a(hidden_states) # [batch_size, seq_len, num_v_heads]

        beta = torch.sigmoid(b)
        g = -torch.exp(self.A_log) * F.softplus(a + self.dt_bias) 

        if self.num_v_heads // self.num_k_heads > 1:
            rep = self.num_v_heads // self.num_k_heads
            q = q.repeat_interleave(rep, dim=2) # [batch_size, seq_len, num_k_heads, head_k_dim]
            k = k.repeat_interleave(rep, dim=2) # [batch_size, seq
 
        attn_core, reccurrent_state = torch_recurrent_gated_delta_rule(
            q, k, v, g,beta,
            reccurrent_state,
            cache if cache is not None else None,
        )

        attn_core = attn_core.reshape(-1, self.v_dim) # [batch_size*seq_len, v_dim]
        z = z.reshape(-1, self.v_dim) # [batch_size*seq_len, v_dim]
        attn_core = attn_core * F.silu(z)
        attn_core= attn_core.reshape(hidden_states.shape[0], hidden_states.shape[1], self.v_dim) # [batch_size, seq_len, v_dim]

        attn_output = self.proj_out(attn_core) # [batch_size, seq_len, hidden_size]
        return attn_output

class MLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Decoder(nn.Module):
    def __init__(
            self,
            config,
            layer_idx,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "Linear_attention":
            self.Linear_attn = GatedDeltaNet(config, layer_idx)
        else:
            self.self_attn = SelfAttention(config, layer_idx)

        self.input_layer_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(self,
                hidden_states: torch.Tensor,
                pos_embeddings: tuple,
                attention_mask: torch.Tensor | None=None,
                cache: dict = None,
                ):
        residual = hidden_states
        hidden_states = self.input_layer_norm(hidden_states)
        if self.layer_type == "Linear_attention":
            attn_output = self.Linear_attn(
                hidden_states, 
                attention_mask=attention_mask, 
                cache=cache,
            )
        else:
            attn_output = self.self_attn(
                hidden_states, 
                pos_embeddings, 
                attention_mask=attention_mask, 
                cache=cache,
            )
        hidden_states = residual + attn_output
        residual = hidden_states
        hidden_states = self.post_attn_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class DynamicCache(nn.Module):
    def __init__(
            self,
            config,
    ):
        super().__init__()
        self.num_layers = config.num_hidden_layers
        self.recurrent_state = [None for _ in range(self.num_layers)]
        self.conv_state = [None for _ in range(self.num_layers)]
        self.k = [None for _ in range(self.num_layers)]
        self.v = [None for _ in range(self.num_layers)]

        self.transformer_layers = [i for i in range(config.num_hidden_layers) if config.layer_types[i] == "full_attention"]

    
    def update(self, k, v, layer_idx: int):
        if self.k[layer_idx] is None:
            self.k[layer_idx] = k
            self.v[layer_idx] = v
        
        else:
            self.k[layer_idx] = torch.cat((self.k[layer_idx], k), dim=-2)
            self.v[layer_idx] = torch.cat((self.v[layer_idx], v), dim=-2)
        
        return self.k[layer_idx], self.v[layer_idx]


    def get_seq_len(self):
        return self.k[self.transformer_layers[0]].shape[-2] if self.k[self.transformer_layers[0]] is not None else 0

    def get_seq_length(self):
        return self.get_seq_len()


class TextModel(nn.Module):
    def __init__(
            self,
            config,
    ):
        super().__init__()
        self.config = config
        self.rope = RoPE(config)
        self.layers = nn.ModuleList([Decoder(config, i) for i in range(config.num_hidden_layers)])
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.cache = DynamicCache(config)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, 
                input_ids,
                positional_embeddings= None,
                attention_mask = None,
                input_embeddings = None,
                cache = None,
                ):
        #input_ids: [batch_size, seq_len]
        #positional_embeddings: tuple of (cos, sin) each of shape [batch_size
        #attention_mask: [batch_size, seq_len]
        #input_embeddings: [batch_size, seq_len, hidden_size]
        if input_embeddings is None:
            hidden_states = self.embed_tokens(input_ids) # [batch_size, seq_len, hidden_size]
        else:
            hidden_states = input_embeddings
        
        batch_size, seq_len, hidden_size = hidden_states.shape
        if cache is None:
            cache = DynamicCache(self.config)
        
        past_seen_tokens = cache.get_seq_len() if cache.get_seq_len() is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_len + past_seen_tokens),
                dtype=hidden_states.dtype, 
                device=hidden_states.device) # [batch_size, seq_len + past_seen_tokens]
        kv_length = seq_len + past_seen_tokens
        casual_mask = build_causal_mask(
            attention_mask=attention_mask,
            batch_size=batch_size,
            query_length=seq_len,
            kv_length=kv_length,
            device=hidden_states.device,
            dtype=hidden_states.dtype,)
        
        pos_ids = torch.arange(past_seen_tokens, past_seen_tokens + seq_len, device=hidden_states.device) # [seq_len]

        
        positional_embeddings = self.rope(hidden_states, pos_ids)
        
        for layer in self.layers:
            mask = casual_mask if layer.layer_type == "full_attention" else attention_mask
            hidden_states = layer(
                hidden_states,
                pos_embeddings=positional_embeddings,
                attention_mask=mask,
                cache=cache,
            )

        output = self.out_proj(hidden_states)
        return output , cache


if __name__ == "__main__":
    from config import config

    norm=RMSNorm(512)
    x=torch.randn(2,3,4,512)
    output=norm(x)
    print(output.shape)

    batch_size=4
    seq_len= 20

    pos_ids=torch.arange(0, seq_len).expand(batch_size, -1)
    rope=RoPE(config)
    out=rope(torch.randn(batch_size,seq_len, 128), pos_ids)
    print(out[0].shape, out[1].shape)

    # test self attention
    self_attn = SelfAttention(config, layer_idx=0)
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    pos_embeddings = torch.randn(batch_size, seq_len, config.hidden_size//config.num_attention_heads) 

    attn_output = self_attn(
        hidden_states, 
        (pos_embeddings, pos_embeddings),)
    print(attn_output.shape)

    # test gated delta net
    gated_delta_net = GatedDeltaNet(config, layer_idx=0)
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    attn_output = gated_delta_net(
        hidden_states, 
        attention_mask=torch.ones(batch_size, seq_len).bool(),)
    print(attn_output.shape)

    decoder = Decoder(config, layer_idx=1)
    out = decoder(
        hidden_states, 
        (pos_embeddings, pos_embeddings), 
        attention_mask=torch.ones(batch_size, seq_len).bool(),
    )
    print(out.shape)  

    textModel = TextModel(config)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    out, cache = textModel(input_ids)
    print(out.shape)