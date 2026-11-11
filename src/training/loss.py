"""
Módulo de Avaliação de Perda e Perplexidade (Entregável 2)
"""
import math
import torch
import torch.nn as nn


def calc_loss_batch(input_batch, target_batch, model, device):
    """Calcula a perda de entropia cruzada para um lote único."""
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = nn.functional.cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten()
    )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """Calcula a perda média sobre um DataLoader inteiro."""
    model.eval()
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")

    n_batches = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))

    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i >= n_batches:
                break
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()

    return total_loss / n_batches


def calc_perplexity(loss_value: float) -> float:
    """Calcula a perplexidade: PPL = exp(Loss)."""
    try:
        return math.exp(loss_value)
    except OverflowError:
        return float("inf")
