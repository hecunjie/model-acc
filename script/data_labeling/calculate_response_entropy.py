"""
Calculate entropy and probability distribution for LLM responses using a specified model.

Inputs:
- CSV file from step_0_llm_response.py with columns: id, input_text, model_reasoning, model_response, is_finished

Outputs:
- entropy_analysis.csv: Contains token-level entropy, probabilities, and token IDs
- entropy_summary.json: Aggregated statistics (mean/median entropy, etc.)
"""

import argparse
import os
import torch
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import List, Tuple, Dict
import torch.nn.functional as F


def load_model_and_tokenizer(model_path: str):
    """Load model and tokenizer"""
    print(f"Loading model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    model_config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=model_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ).eval()
    
    print(f"Model loaded successfully on {model.device}")
    return model, tokenizer


def calculate_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Calculate entropy for a batch of logits.
    
    Args:
        logits: Tensor of shape [batch_size, vocab_size] or [vocab_size]
    
    Returns:
        Entropy values for each position
    """
    # Convert logits to probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Calculate entropy: -sum(p * log(p))
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    
    return entropy


def get_top_p_tokens(logits: torch.Tensor, top_p: float = 0.9, top_k: int = 50) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get top-p tokens and their probabilities.
    
    Args:
        logits: Tensor of shape [vocab_size]
        top_p: Cumulative probability threshold
        top_k: Maximum number of tokens to consider
    
    Returns:
        token_ids: Token IDs in the top-p set
        probs: Probabilities of those tokens
        cumulative_probs: Cumulative probabilities
    """
    # Convert to probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Sort probabilities
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    # Calculate cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Find where cumulative probability exceeds top_p
    sorted_indices_to_keep = cumulative_probs <= top_p
    
    # Always keep at least the top token
    sorted_indices_to_keep[0] = True
    
    # Apply top_k limit if specified
    if top_k > 0:
        sorted_indices_to_keep[top_k:] = False
    
    # Get the tokens and probabilities to keep
    kept_probs = sorted_probs[sorted_indices_to_keep]
    kept_indices = sorted_indices[sorted_indices_to_keep]
    kept_cumulative = cumulative_probs[sorted_indices_to_keep]
    
    return kept_indices, kept_probs, kept_cumulative


def apply_chat_template_with_response(tokenizer, input_text: str, model_reasoning: str, model_response: str, model_name: str) -> str:
    """
    Apply chat template to reconstruct the full conversation including response.
    
    Args:
        tokenizer: Tokenizer instance
        input_text: Original user input
        model_reasoning: Reasoning content (inside <think> tags)
        model_response: Final response content
        model_name: Name of the model (to check for R1 models)
    
    Returns:
        Full formatted prompt
    """
    # Check if this is an R1 model
    is_r1_model = 'r1' in model_name.lower()
    
    messages = [
        {"role": "user", "content": input_text},
        {"role": "assistant", "content": None}
    ]
    
    if is_r1_model:
        # R1 models use <think> tags
        messages[1]["content"] = f"<think>\n{model_reasoning}\n</think>\n\n{model_response}"
    else:
        # Other models might use different formats
        messages[1]["content"] = f"{model_reasoning}\n</think>\n\n{model_response}"
    
    # Apply chat template
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
    except Exception as e:
        print(f"Warning: Failed to apply chat template: {e}")
        # Fallback: simple concatenation
        prompt = f"User: {input_text}\nAssistant: {messages[1]['content']}"
    
    return prompt


def analyze_response(
    model,
    tokenizer,
    sample: Dict,
    model_name: str,
    top_p: float = 0.9,
    top_k: int = 50
) -> List[Dict]:
    """
    Analyze a single response and calculate entropy and probability distributions.
    
    Returns:
        List of dictionaries containing token-level analysis
    """
    # Reconstruct the full prompt with response
    input_text = sample["input_text"]
    model_reasoning = sample["model_reasoning"]
    model_response = sample["model_response"]
    
    if not model_reasoning or not model_response:
        print(f"Skipping sample {sample['id']}: Missing reasoning or response")
        return []
    
    # Get the full formatted prompt
    full_prompt = apply_chat_template_with_response(
        tokenizer, input_text, model_reasoning, model_response, model_name
    )
    
    # Tokenize
    input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to(model.device)
    
    # Skip if too long
    if input_ids.shape[1] > 32768:
        print(f"Skipping sample {sample['id']}: Too long ({input_ids.shape[1]} tokens)")
        return []
    
    # Run model inference
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0]  # Shape: [seq_len, vocab_size]
    
    # Calculate entropy for each position
    entropies = calculate_entropy(logits).cpu().numpy()
    
    # Decode tokens
    tokens = input_ids[0].cpu().tolist()
    token_texts = [tokenizer.decode([t]) for t in tokens]
    
    # Analyze each position
    results = []
    for pos in range(len(tokens)):
        # Get logits for this position
        pos_logits = logits[pos]
        
        # Get top-p tokens
        top_p_token_ids, top_p_probs, top_p_cumulative = get_top_p_tokens(
            pos_logits, top_p=top_p, top_k=top_k
        )
        
        # Convert to lists
        top_p_token_ids = top_p_token_ids.cpu().tolist()
        top_p_probs = top_p_probs.cpu().tolist()
        top_p_cumulative = top_p_cumulative.cpu().tolist()
        
        # Get the actual next token (if exists)
        actual_token_id = tokens[pos + 1] if pos + 1 < len(tokens) else None
        actual_token_prob = None
        if actual_token_id is not None:
            # Get probability of actual next token
            actual_token_prob = F.softmax(pos_logits, dim=-1)[actual_token_id].item()
        
        result = {
            "sample_id": sample["id"],
            "position": pos,
            "token_id": tokens[pos],
            "token_text": token_texts[pos],
            "entropy": float(entropies[pos]),
            "actual_next_token_id": actual_token_id,
            "actual_next_token_prob": actual_token_prob,
            "top_p_token_ids": top_p_token_ids,
            "top_p_probs": top_p_probs,
            "top_p_cumulative_probs": top_p_cumulative,
            "top_p_size": len(top_p_token_ids)
        }
        
        results.append(result)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Calculate entropy and probability distributions for LLM responses"
    )
    
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to the CSV file from step_0_llm_response.py"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model for calculating entropy"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="entropy_analysis",
        help="Directory to save output files"
    )
    
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p threshold for probability distribution"
    )
    
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k limit for probability distribution"
    )
    
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to process (for testing)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load CSV
    print(f"Loading CSV from {args.csv_path}...")
    df = pd.read_csv(args.csv_path)
    print(f"Loaded {len(df)} samples")
    
    # Filter to only finished samples
    df = df[df["is_finished"] == True]
    print(f"Filtered to {len(df)} finished samples")
    
    # Limit number of samples if specified
    if args.num_samples:
        df = df.head(args.num_samples)
        print(f"Processing first {args.num_samples} samples")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    
    # Process each sample
    all_results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing responses"):
        sample = row.to_dict()
        
        results = analyze_response(
            model=model,
            tokenizer=tokenizer,
            sample=sample,
            model_name=args.model_path,
            top_p=args.top_p,
            top_k=args.top_k
        )
        
        all_results.extend(results)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save detailed results
    output_csv = os.path.join(args.output_dir, "entropy_analysis.csv")
    results_df.to_csv(output_csv, index=False)
    print(f"Saved detailed analysis to {output_csv}")
    
    # Calculate summary statistics
    summary = {
        "model_path": args.model_path,
        "num_samples": len(df),
        "total_tokens": len(results_df),
        "mean_entropy": float(results_df["entropy"].mean()),
        "median_entropy": float(results_df["entropy"].median()),
        "std_entropy": float(results_df["entropy"].std()),
        "min_entropy": float(results_df["entropy"].min()),
        "max_entropy": float(results_df["entropy"].max()),
        "mean_top_p_size": float(results_df["top_p_size"].mean()),
        "mean_actual_token_prob": float(results_df["actual_next_token_prob"].dropna().mean()),
        "top_p": args.top_p,
        "top_k": args.top_k
    }
    
    # Save summary
    summary_json = os.path.join(args.output_dir, "entropy_summary.json")
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary statistics to {summary_json}")
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("="*50)


if __name__ == "__main__":
    main()
