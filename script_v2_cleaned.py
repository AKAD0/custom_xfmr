import torch
import torch.nn as nn
from torch.nn import functional as F 

# ------ Hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_interval = 300
eval_iters = 200
torch.manual_seed(1337)
n_embd = 32

# ------ Preparing dataset
# Loading dataset
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted( list( set( text)))
vocab_size = len( chars)

# Tokenization
stoi = { ch:i for i,ch in enumerate( chars) }
itos = { i:ch for i,ch in enumerate( chars) }
encode = lambda s: [ stoi[c] for c in s]
decode = lambda l: ''.join( [ itos[i] for i in l])
data = torch.tensor( encode(text), dtype=torch.long)    #Tokenizing dataset

# Splitting dataset
n = int( 0.9*len( data))
train_data = data[:n]
val_data = data[n:]

# Batching and loading data
def get_batch(split):
    data = train_data if split == 'train' else val_data         #selecting type of split
    ix = torch.randint( len(data)-block_size, (batch_size,))    #selecting random positions to pull blocks from
    x = torch.stack( [data[i:i+block_size] for i in ix])        # }concatenation of blocks in a batch
    y = torch.stack( [data[i+1:i+block_size+1] for i in ix])    # }
    x, y = x.to(device), y.to(device)                           #loading data to device
    return x, y

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

# ------ Architecture
class BigramLanguageModel(nn.Module):
    def __init__( self):
        super().__init__()
        self.token_embedding_table = nn.Embedding( vocab_size, n_embd)              #creating an embeddings tensor (it's an object, not a collection)
        self.position_embedding_table = nn.Embedding( block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward( self, idx, targets=None):                                          #'targets=None" means this argument is optional
        B,T = idx.shape
        tok_emb = self.token_embedding_table( idx)                                  #(B,T,C) where C = n_embd
        pos_emb = self.position_embedding_table( torch.arange( T, device=device))   #(T,C)
        x = tok_emb + pos_emb                                                       #(B,T,C)
        logits = self.lm_head( x)                                                   #(B,T,C) where C = vocab_size

        if targets is None:                                                 
            loss = None
        else:
            B,T,C = logits.shape                                                    # }'F.cross_entropy()' takes R^2 dimensions. So B,T,C is converted to B*T,C
            logits = logits.view( B*T, C)                                           # } 
            targets = targets.view( B*T)                                            # }
            loss = F.cross_entropy( logits, targets)                                # }

        return logits, loss

    def generate( self, idx, max_new_tokens):
        for _ in range( max_new_tokens):
            logits, loss = self(idx)                                        #generate predictions utilizing the 'nn.Embedding' tensor contents. 'loss' will be ignored
            logits = logits[ :, -1, :]                                      #take only the data for last token
            probs = F.softmax( logits, dim=1)                               #mapping predictions into probabilities via softmax()
            idx_next = torch.multinomial( probs, num_samples=1)             #taking only the highest probability
            idx = torch.cat( ( idx, idx_next), dim=1)                       #concatenate the prediction to the current context
        return idx

model = BigramLanguageModel()    #creating model object
m = model.to(device)                        #passing model (parameters) to the device

# ------ Optimization
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

# ------ Inference
context = torch.zeros( (1,1), dtype=torch.long, device=device)
#print( decode( m.generate( context, max_new_tokens=500)[0].tolist()))      #!!!after adding pos encoding it throws an error