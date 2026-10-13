"""
Configurações do modelo GPT
===========================
GPT_CONFIG_124M reproduz o GPT-2 "small" do livro (cap. 4.1).

GPT_CONFIG_MINI é uma versão reduzida para testes e experimentos rápidos
em CPU, com a mesma arquitetura e o mesmo vocabulário BPE.

Chaves opcionais aceitas pelos módulos (usadas nos experimentos):
    ff_mult             fator de expansão da FeedForward (padrão 4)
    drop_rate_emb       dropout após os embeddings       (padrão drop_rate)
    drop_rate_attn      dropout nos pesos de atenção     (padrão drop_rate)
    drop_rate_shortcut  dropout antes das residuais      (padrão drop_rate)
"""

GPT_CONFIG_124M = {
    "vocab_size": 50257,     # vocabulário do tokenizador BPE do GPT-2
    "context_length": 1024,  # máximo de tokens por entrada
    "emb_dim": 768,          # dimensão dos embeddings
    "n_heads": 12,           # cabeças de atenção
    "n_layers": 12,          # Transformer Blocks empilhados
    "drop_rate": 0.1,        # taxa de dropout
    "qkv_bias": False,       # bias nas projeções Q, K, V
}

GPT_CONFIG_MINI = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 128,
    "n_heads": 4,
    "n_layers": 2,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

# Exercício 4.2: variantes maiores do GPT-2 (mudam só emb_dim, n_layers e n_heads)
GPT2_VARIANTS = {
    "gpt2-small (124M)":  {"emb_dim": 768,  "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)":  {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)":    {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
