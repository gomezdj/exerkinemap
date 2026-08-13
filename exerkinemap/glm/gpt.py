import torch
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

class MoTrPACSequenceGenerator:
    def __init__(self, vocab_size=50265, max_length=1024, model_path=None):
        """
        Initializes a GPT-style autoregressive model for sequence generation.
        """
        # GPT configuration tailored for biological sequence generation
        self.config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=max_length,
            n_embd=768,      # Embedding dimension
            n_layer=12,      # Number of transformer blocks
            n_head=12,       # Number of attention heads
            bos_token_id=0,
            eos_token_id=2,
        )
        
        # Load pre-trained foundation model weights (e.g., Omni-DNA) or initialize from scratch
        if model_path:
            self.model = GPT2LMHeadModel.from_pretrained(model_path)
        else:
            self.model = GPT2LMHeadModel(self.config)
            
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def generate(self, tokenizer, prompt_sequence, max_new_tokens=150, temperature=0.7):
        """
        Autoregressively generates genomic or exerkine sequences from a prompt.
        """
        self.model.eval()
        
        # Encode prompt using your custom biological tokenizer
        input_ids = tokenizer.encode(prompt_sequence, return_tensors="pt").to(self.device)
        
        # Generate sequence
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,          # Enable stochastic sampling
                top_k=50,                # Filter to top 50 tokens
                top_p=0.95,              # Nucleus sampling
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            )
            
        # Decode tensor back into sequence string
        generated_sequence = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_sequence

def generate(self, tokenizer, prompt_sequence, max_new_tokens=150, temperature=0.7, as_rna=False):
        """
        Autoregressively generates genomic or exerkine sequences from a prompt.
        """
        self.model.eval()
        
        # If the user provides an RNA prompt but the model expects DNA, 
        # temporarily convert it to DNA for tokenization.
        if as_rna:
            prompt_sequence = prompt_sequence.replace("U", "T")
        
        # Encode prompt using your custom biological tokenizer
        input_ids = tokenizer.encode(prompt_sequence, return_tensors="pt").to(self.device)
        
        # Generate sequence
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,          
                top_k=50,                
                top_p=0.95,              
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            )
            
        # Decode tensor back into sequence string
        generated_sequence = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Transcribe DNA to RNA if requested
        if as_rna:
            generated_sequence = generated_sequence.replace("T", "U")
            
        return generated_sequence

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # 1. Load the tokenizer trained on the 177 unique genomic targets
    # (Assuming you saved it as a standard fast tokenizer JSON)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="bio-gpt-tokenizer.json")
    tokenizer.pad_token = "[PAD]"
    
    # 2. Initialize the generator
    generator = MoTrPACSequenceGenerator(vocab_size=tokenizer.vocab_size)
    
    # 3. Provide a starting k-mer or sequence chunk
    prompt = "ATG CGT ACG"
    
    # 4. Generate the novel sequence
    new_sequence = generator.generate(tokenizer, prompt_sequence=prompt, max_new_tokens=50)
    print(f"Generated Sequence: {new_sequence}")


    # You can now provide a prompt with 'U' or 'T'
    prompt = "AUG CGU ACG" 
    
    # Generate the novel sequence as RNA
    new_sequence = generator.generate(
        tokenizer, 
        prompt_sequence=prompt, 
        max_new_tokens=50, 
        as_rna=True  # <--- Set this to True
    )
    
    print(f"Generated RNA Sequence: {new_sequence}")