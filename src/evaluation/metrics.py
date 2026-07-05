"""
Metrics for evaluating Pragmatic Drift (δp) in RAG systems.
"""

import torch
import torch.nn.functional as F
import numpy as np


def calculate_jsd(base_probs, pert_probs, top_k=10):
    """
    Calculate Jensen-Shannon Divergence between two probability distributions.
    
    Args:
        base_probs (torch.Tensor): Baseline probability distribution
        pert_probs (torch.Tensor): Perturbed probability distribution
        top_k (int): Number of top tokens to consider (to reduce noise)
    
    Returns:
        float: Jensen-Shannon Divergence value
    """
    # Restrict to top_k tokens
    top_k_vals, top_k_indices = torch.topk(base_probs, top_k)
    base_top = base_probs[top_k_indices]
    pert_top = pert_probs[top_k_indices]
    
    # Normalize
    base_top = base_top / base_top.sum()
    pert_top = pert_top / pert_top.sum()
    
    # Mixture distribution
    M = 0.5 * (base_top + pert_top)
    
    # KL divergences
    kl_base = F.kl_div(torch.log(base_top + 1e-10), M, reduction='batchmean')
    kl_pert = F.kl_div(torch.log(pert_top + 1e-10), M, reduction='batchmean')
    
    # Jensen-Shannon Divergence
    jsd = 0.5 * (kl_base + kl_pert)
    
    return jsd.item()


def calculate_delta_p(model, tokenizer, query, doc, instruction, 
                      perturbations, top_tokens=10):
    """
    Calculate Pragmatic Drift (δp) for a given context.
    
    Args:
        model: HuggingFace causal LM
        tokenizer: Corresponding tokenizer
        query (str): User query
        doc (str): Retrieved document chunk
        instruction (str): System instruction
        perturbations (list): List of perturbed instructions
        top_tokens (int): Number of top tokens for JSD calculation
    
    Returns:
        dict: {'drift': avg_jsd, 'perplexity_shift': avg_perp_shift}
    """
    # Baseline
    baseline_prompt = f"{instruction}\n\nDocument: {doc}\n\nQuery: {query}"
    base_inputs = tokenizer(baseline_prompt, return_tensors="pt")
    
    with torch.no_grad():
        base_outputs = model(**base_inputs, output_logits=True)
        base_logits = base_outputs.logits[0, -1, :]
        base_probs = F.softmax(base_logits, dim=-1)
        base_entropy = -torch.sum(base_probs * torch.log(base_probs + 1e-10))
    
    drift_scores = []
    perp_shifts = []
    
    for pert in perturbations:
        perturbed_prompt = f"{pert}\n\nDocument: {doc}\n\nQuery: {query}"
        pert_inputs = tokenizer(perturbed_prompt, return_tensors="pt")
        
        with torch.no_grad():
            pert_outputs = model(**pert_inputs, output_logits=True)
            pert_logits = pert_outputs.logits[0, -1, :]
            pert_probs = F.softmax(pert_logits, dim=-1)
            pert_entropy = -torch.sum(pert_probs * torch.log(pert_probs + 1e-10))
        
        jsd = calculate_jsd(base_probs, pert_probs, top_k=top_tokens)
        drift_scores.append(jsd)
        perp_shifts.append((base_entropy - pert_entropy).item())
    
    return {
        'drift': np.mean(drift_scores),
        'perplexity_shift': np.mean(perp_shifts)
    }


def p2a_stress_test(model, tokenizer, query, doc, instruction,
                    perturbations, threshold=0.5, top_tokens=10):
    """
    Run the P2A stress-test guardrail.
    
    Returns:
        dict: {'status': 'PASS' | 'REJECT', 'drift': float, 'perplexity_shift': float}
    """
    result = calculate_delta_p(model, tokenizer, query, doc, instruction,
                               perturbations, top_tokens)
    
    if result['drift'] > threshold:
        return {'status': 'REJECT', **result}
    else:
        return {'status': 'PASS', **result}