# SPDX-License-Identifier: Apache-2.0

import os
import shutil

import torch
import torch.distributed as dist

from .factory import register_algorithm
from .template import AlgorithmTemplate


@register_algorithm("capture_acts")
class CaptureActs(AlgorithmTemplate):
    """Simple algorithm that captures activations during inference by appending to layer files"""

    def _transform(self, hidden_states: torch.Tensor, params: dict) -> torch.Tensor:
        """
        Capture hidden states by appending to layer-specific file.

        Only rank 0 saves to avoid redundancy (all ranks have full activations after gather).

        Args:
            hidden_states: [num_tokens, hidden_dim] tensor
            params: dict with "save_dir" and "layer_id" keys
        """
        try:
            save_dir = params["save_dir"]
            layer_id = params["layer_id"]

            # Get rank for tensor parallel workers (0 if not using TP)
            if dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0

            # Only rank 0 saves (all ranks have identical full activations after gather)
            if rank != 0:
                return hidden_states

            file_path = os.path.join(save_dir, f"layer_{layer_id}_acts.pt")

            # Move to CPU and detach to avoid GPU memory issues and gradients
            hidden_states_cpu = hidden_states.detach().cpu()

            # Load existing activations if file exists, otherwise start fresh
            if os.path.exists(file_path):
                existing_acts = torch.load(file_path, map_location='cpu', weights_only=False)
                # Concatenate new activations to existing ones
                combined_acts = torch.cat([existing_acts, hidden_states_cpu], dim=0)
            else:
                combined_acts = hidden_states_cpu

            # Save combined activations
            torch.save(combined_acts, file_path)

        except Exception as e:
            # Log error but don't crash inference
            print(f"[CaptureActs] Error saving activations for layer {params.get('layer_id', 'unknown')}: {e}")

        return hidden_states

    @classmethod
    def load_from_path(cls, path: str, device: str, **kwargs) -> dict:
        """
        Initialize capture directory and create layer configs.

        Args:
            path: Directory where layer activation files will be saved
            device: Device (not used for this algorithm)
            **kwargs: Must contain 'target_layers' list

        Returns:
            dict: {"layer_payloads": {layer_id: config_dict}} for each target layer
        """
        # Get rank to coordinate directory operations across tensor parallel workers
        if dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0

        # Only rank 0 manages the directory to avoid race conditions
        if rank == 0:
            # Wipe directory if it exists, then create fresh
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)

        # Synchronize all ranks after directory setup
        if dist.is_initialized():
            dist.barrier()

        # Get target layers from kwargs
        target_layers = kwargs.get("target_layers", [])
        if not target_layers:
            raise ValueError("CaptureActs requires 'target_layers' in kwargs")

        # Create config for each layer with its layer_id
        layer_payloads = {}
        for layer_id in target_layers:
            layer_payloads[layer_id] = {
                "save_dir": path,
                "layer_id": layer_id
            }

        return {"layer_payloads": layer_payloads}
