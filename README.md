# Contents:
1. Describtion
2. Problem
3. Solution
    * 3.1. Estimation;
    * 3.2. Tokenization;
    * 3.3. Single Head Attention;
    * 3.4. Multi-head Attention;
    * 3.5. Feed Forward Layer;
    * 3.6. Optimizer.
4. Results

# 1. Describtion
A simple Transformer written on PyTorch.

# 2. Problem
Transformer architecture implements several blocks:
1. Estimation;
2. Tokenization;
3. Single Head Attention;
4. Multi-head Attention;
5. Feed Forward Layer;
6. Optimizer.
Develop an inferencable architecture with every block mentioned.

# 3. Solution
Solution is comprised of 3 parts: EDA + feature Engineering, Training classifier, Clustering. Describtion of every part is provided below.
## 3.1. Estimation
The code for Estimation block:
```
# ------ Estimation //Mean accuracy for both train and val splits.
@torch.no_grad()                            #disabling back propagation for this fucntion
def estimate_loss():
    out = {}
    model.eval()                            #switching the model to the evaluation mode
    for split in ['train','val']:
        losses = torch.zeros( eval_iters)   #setting up a tensor to store the evaluations
        for k in range( eval_iters):
            X, Y = get_batch( split)
            logits, loss = model( X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()                           #switching model back to the train mode
    return out
```

## 3.2. Tokenization
The code for Tokenization block:
```
# Tokenization
stoi = { ch:i for i,ch in enumerate( chars) }
itos = { i:ch for i,ch in enumerate( chars) }
encode = lambda s: [ stoi[c] for c in s]
decode = lambda l: ''.join( [ itos[i] for i in l])
data = torch.tensor( encode(text), dtype=torch.long)    #Tokenizing dataset
```

## 3.3. Single Head Attention
The code for Single Head Attention block:
```
class Head( nn.Module):
    def __init__( self, head_size):
        super().__init__()
        self.key = nn.Linear( n_embd, head_size, bias=False)
        self.query = nn.Linear( n_embd, head_size, bias=False)
        self.value = nn.Linear( n_embd, head_size, bias=False)
        self.register_buffer( 'tril', torch.tril( torch.ones( block_size, block_size)))    #'tril' isn't a param of the model.
                                                                                           #Thus it has to be made for pytorch as 'buffer'
    def forward( self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * C**-0.5                                      #1/sqrt(C) is a normalization
        wei = wei.masked_fill( self.tril[:T, :T]==0, float('-inf'))                 #switching every '0' with '-inf' so tokens communicate only with previous ones
        wei = F.softmax(wei, dim=-1)                                                #normalizing with softmax
        v = self.value(x)
        out = wei @ v
        return out
```

## 3.4. Multi-head Attention
The code for Multi-head Attention block:
```
class MultiHeadAttention( nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList( [Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear( n_embd, n_embd)

    def forward( self,x):
        out = torch.cat( [h(x) for h in self.heads], dim=-1)
        out = self.proj( out)
        return out
```

## 3.5. Feed Forward Layer
The code for Feed Forward Layer block:
```
class FeedForward( nn.Module):
    def __init__( self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear( n_embd, 4*n_embd),           # } '4 *' widens the layer to increase the capacity of the model
            nn.ReLU(),                              # }
            nn.Linear( 4*n_embd, n_embd),           # }
        )

    def forward( self, x):
        return self.net(x)
```

## 3.6. Optimizer
The code for Optimizer block:
```
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
for iter in range(max_iters):
    if iter % eval_interval ==0:                                                                #do evaluation every 'eval_interval' iteration
        losses = estimate_loss()
        print( f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch( 'train')
    logits, loss = model( xb, yb)
    optimizer.zero_grad( set_to_none=True)
    loss.backward()
    optimizer.step()
```
# 5. Results
Resulting transformer was taught on Shakespear's texts and was able to learn general patterns in it's body, such as the common length of a word, the script-like nature of the text and some specific words.
<p align="center">
  <img src="https://github.com/AKAD0/custom_xfmr/blob/master/results.jpg">
</p>

$$
\text{Fig.1: Transformer learned general patterns of the text}
$$


The following tasks have been performed:

1. Developed an inferencable architecture with every needed blocks mentioned.
2. Transformer has been taught to generate text based on Shakespear scripts.
