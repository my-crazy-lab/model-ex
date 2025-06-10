"""
Model quantization implementations
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import os
import tempfile
from pathlib import Path

from ..config.quantization_config import QuantizationConfig, QuantizationType, QuantizationBackend

logger = logging.getLogger(__name__)


class ModelQuantizer:
    """
    Main model quantizer that supports multiple quantization backends and methods
    """
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
        
        # Initialize backend-specific quantizer
        if config.backend == QuantizationBackend.PYTORCH:
            self.quantizer = PyTorchQuantizer(config)
        elif config.backend == QuantizationBackend.BITSANDBYTES:
            self.quantizer = BitsAndBytesQuantizer(config)
        elif config.backend == QuantizationBackend.NEURAL_COMPRESSOR:
            self.quantizer = NeuralCompressorQuantizer(config)
        elif config.backend == QuantizationBackend.ONNX:
            self.quantizer = ONNXQuantizer(config)
        else:
            raise ValueError(f"Unsupported backend: {config.backend}")
        
        logger.info(f"ModelQuantizer initialized with {config.backend.value} backend")
    
    def quantize(
        self,
        model: nn.Module,
        calibration_dataloader: Optional[Any] = None,
        example_inputs: Optional[torch.Tensor] = None
    ) -> nn.Module:
        """
        Quantize the model using the configured backend
        
        Args:
            model: Model to quantize
            calibration_dataloader: Data for calibration (if needed)
            example_inputs: Example inputs for tracing (if needed)
            
        Returns:
            Quantized model
        """
        logger.info("Starting model quantization...")
        
        # Validate inputs
        if self.config.quantization_type == QuantizationType.STATIC and calibration_dataloader is None:
            raise ValueError("Static quantization requires calibration data")
        
        # Quantize model
        quantized_model = self.quantizer.quantize(
            model=model,
            calibration_dataloader=calibration_dataloader,
            example_inputs=example_inputs
        )
        
        logger.info("Model quantization completed")
        return quantized_model
    
    def get_model_size(self, model: nn.Module) -> int:
        """Get model size in bytes"""
        return self.quantizer.get_model_size(model)
    
    def benchmark_model(
        self,
        model: nn.Module,
        example_inputs: torch.Tensor,
        num_runs: int = 100
    ) -> Dict[str, float]:
        """Benchmark model performance"""
        return self.quantizer.benchmark_model(model, example_inputs, num_runs)
    
    def save_quantized_model(self, model: nn.Module, save_path: str):
        """Save quantized model"""
        self.quantizer.save_quantized_model(model, save_path)


class PyTorchQuantizer:
    """PyTorch native quantization implementation"""
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
        
    def quantize(
        self,
        model: nn.Module,
        calibration_dataloader: Optional[Any] = None,
        example_inputs: Optional[torch.Tensor] = None
    ) -> nn.Module:
        """Quantize model using PyTorch quantization"""
        
        if self.config.quantization_type == QuantizationType.DYNAMIC:
            return self._dynamic_quantization(model)
        elif self.config.quantization_type == QuantizationType.STATIC:
            return self._static_quantization(model, calibration_dataloader)
        elif self.config.quantization_type in [QuantizationType.FP16, QuantizationType.BF16]:
            return self._mixed_precision_quantization(model)
        else:
            raise ValueError(f"Unsupported quantization type: {self.config.quantization_type}")
    
    def _dynamic_quantization(self, model: nn.Module) -> nn.Module:
        """Apply dynamic quantization"""
        logger.info("Applying dynamic quantization...")
        
        # Specify layers to quantize
        layers_to_quantize = {nn.Linear}
        if self.config.quantize_embeddings:
            layers_to_quantize.add(nn.Embedding)
        
        # Apply dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            layers_to_quantize,
            dtype=torch.qint8 if self.config.bits == 8 else torch.qint4
        )
        
        return quantized_model
    
    def _static_quantization(self, model: nn.Module, calibration_dataloader: Any) -> nn.Module:
        """Apply static quantization with calibration"""
        logger.info("Applying static quantization...")
        
        # Set quantization configuration
        model.qconfig = self.config.get_pytorch_qconfig()
        
        # Prepare model for quantization
        model_prepared = torch.quantization.prepare(model, inplace=False)
        
        # Calibration
        logger.info("Running calibration...")
        model_prepared.eval()
        
        with torch.no_grad():
            for i, batch in enumerate(calibration_dataloader):
                if i >= self.config.num_calibration_samples // self.config.calibration_batch_size:
                    break
                
                # Move batch to appropriate device
                if isinstance(batch, dict):
                    batch = {k: v.to(model_prepared.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    _ = model_prepared(**batch)
                else:
                    if isinstance(batch, (list, tuple)):
                        batch = batch[0]  # Take first element if tuple/list
                    batch = batch.to(model_prepared.device)
                    _ = model_prepared(batch)
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(model_prepared, inplace=False)
        
        return quantized_model
    
    def _mixed_precision_quantization(self, model: nn.Module) -> nn.Module:
        """Apply mixed precision (FP16/BF16)"""
        logger.info(f"Applying {self.config.quantization_type.value} mixed precision...")
        
        if self.config.quantization_type == QuantizationType.FP16:
            return model.half()
        elif self.config.quantization_type == QuantizationType.BF16:
            return model.to(torch.bfloat16)
        
        return model
    
    def get_model_size(self, model: nn.Module) -> int:
        """Get model size in bytes"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        return param_size + buffer_size
    
    def benchmark_model(
        self,
        model: nn.Module,
        example_inputs: torch.Tensor,
        num_runs: int = 100
    ) -> Dict[str, float]:
        """Benchmark model performance"""
        import time
        
        model.eval()
        device = next(model.parameters()).device
        example_inputs = example_inputs.to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(example_inputs)
        
        # Benchmark
        torch.cuda.synchronize() if device.type == 'cuda' else None
        
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(example_inputs)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        throughput = 1.0 / avg_time
        
        return {
            "avg_inference_time": avg_time,
            "throughput": throughput,
            "total_time": end_time - start_time
        }
    
    def save_quantized_model(self, model: nn.Module, save_path: str):
        """Save quantized model"""
        os.makedirs(save_path, exist_ok=True)
        
        # Save model state dict
        torch.save(model.state_dict(), os.path.join(save_path, "quantized_model.pt"))
        
        # Save full model
        torch.save(model, os.path.join(save_path, "quantized_model_full.pt"))
        
        logger.info(f"Quantized model saved to {save_path}")


class BitsAndBytesQuantizer:
    """BitsAndBytes quantization implementation"""
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
        
        try:
            import bitsandbytes as bnb
            self.bnb = bnb
        except ImportError:
            raise ImportError("BitsAndBytes not available. Install with: pip install bitsandbytes")
    
    def quantize(
        self,
        model: nn.Module,
        calibration_dataloader: Optional[Any] = None,
        example_inputs: Optional[torch.Tensor] = None
    ) -> nn.Module:
        """Quantize model using BitsAndBytes"""
        logger.info("Applying BitsAndBytes quantization...")
        
        # For transformers models, quantization is applied during loading
        # For other models, we need to replace linear layers
        if hasattr(model, 'config') and hasattr(model.config, 'model_type'):
            # This is a transformers model - quantization should be applied during loading
            logger.warning("BitsAndBytes quantization for transformers models should be applied during model loading")
            return model
        else:
            # Replace linear layers with quantized versions
            return self._replace_linear_layers(model)
    
    def _replace_linear_layers(self, model: nn.Module) -> nn.Module:
        """Replace linear layers with quantized versions"""
        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                # Replace with quantized linear layer
                if self.config.load_in_8bit:
                    quantized_layer = self.bnb.nn.Linear8bitLt(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None
                    )
                elif self.config.load_in_4bit:
                    quantized_layer = self.bnb.nn.Linear4bit(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None,
                        compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
                        quant_type=self.config.bnb_4bit_quant_type
                    )
                else:
                    continue
                
                # Copy weights and bias
                quantized_layer.weight.data = module.weight.data
                if module.bias is not None:
                    quantized_layer.bias.data = module.bias.data
                
                setattr(model, name, quantized_layer)
            else:
                # Recursively replace in child modules
                self._replace_linear_layers(module)
        
        return model
    
    def get_model_size(self, model: nn.Module) -> int:
        """Get model size in bytes"""
        total_size = 0
        
        for param in model.parameters():
            if hasattr(param, 'quant_state'):
                # Quantized parameter
                if param.quant_state.dtype == torch.uint8:
                    total_size += param.numel()  # 8-bit
                else:
                    total_size += param.numel() // 2  # 4-bit
            else:
                # Regular parameter
                total_size += param.nelement() * param.element_size()
        
        return total_size
    
    def benchmark_model(
        self,
        model: nn.Module,
        example_inputs: torch.Tensor,
        num_runs: int = 100
    ) -> Dict[str, float]:
        """Benchmark model performance"""
        # Use same benchmarking as PyTorch
        pytorch_quantizer = PyTorchQuantizer(self.config)
        return pytorch_quantizer.benchmark_model(model, example_inputs, num_runs)
    
    def save_quantized_model(self, model: nn.Module, save_path: str):
        """Save quantized model"""
        os.makedirs(save_path, exist_ok=True)
        
        # Save model
        torch.save(model.state_dict(), os.path.join(save_path, "quantized_model_bnb.pt"))
        
        logger.info(f"BitsAndBytes quantized model saved to {save_path}")


class NeuralCompressorQuantizer:
    """Neural Compressor quantization implementation"""
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
        
        try:
            from neural_compressor import quantization
            self.nc_quantization = quantization
        except ImportError:
            raise ImportError("Neural Compressor not available. Install with: pip install neural-compressor")
    
    def quantize(
        self,
        model: nn.Module,
        calibration_dataloader: Optional[Any] = None,
        example_inputs: Optional[torch.Tensor] = None
    ) -> nn.Module:
        """Quantize model using Neural Compressor"""
        logger.info("Applying Neural Compressor quantization...")
        
        # Get quantization configuration
        conf = self.config.get_neural_compressor_config()
        
        # Apply quantization
        quantized_model = self.nc_quantization.fit(
            model=model,
            conf=conf,
            calib_dataloader=calibration_dataloader
        )
        
        return quantized_model
    
    def get_model_size(self, model: nn.Module) -> int:
        """Get model size in bytes"""
        # Use PyTorch method as fallback
        pytorch_quantizer = PyTorchQuantizer(self.config)
        return pytorch_quantizer.get_model_size(model)
    
    def benchmark_model(
        self,
        model: nn.Module,
        example_inputs: torch.Tensor,
        num_runs: int = 100
    ) -> Dict[str, float]:
        """Benchmark model performance"""
        pytorch_quantizer = PyTorchQuantizer(self.config)
        return pytorch_quantizer.benchmark_model(model, example_inputs, num_runs)
    
    def save_quantized_model(self, model: nn.Module, save_path: str):
        """Save quantized model"""
        os.makedirs(save_path, exist_ok=True)
        
        # Save model
        torch.save(model, os.path.join(save_path, "quantized_model_nc.pt"))
        
        logger.info(f"Neural Compressor quantized model saved to {save_path}")


class ONNXQuantizer:
    """ONNX quantization implementation"""
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
        
        try:
            import onnx
            import onnxruntime
            from onnxruntime.quantization import quantize_dynamic, quantize_static, CalibrationDataReader
            
            self.onnx = onnx
            self.onnxruntime = onnxruntime
            self.quantize_dynamic = quantize_dynamic
            self.quantize_static = quantize_static
            self.CalibrationDataReader = CalibrationDataReader
        except ImportError:
            raise ImportError("ONNX/ONNXRuntime not available. Install with: pip install onnx onnxruntime")
    
    def quantize(
        self,
        model: nn.Module,
        calibration_dataloader: Optional[Any] = None,
        example_inputs: Optional[torch.Tensor] = None
    ) -> str:
        """Quantize model using ONNX (returns path to quantized ONNX model)"""
        logger.info("Applying ONNX quantization...")
        
        if example_inputs is None:
            raise ValueError("ONNX quantization requires example inputs for model export")
        
        # Export to ONNX
        with tempfile.TemporaryDirectory() as temp_dir:
            onnx_path = os.path.join(temp_dir, "model.onnx")
            quantized_path = os.path.join(self.config.output_dir, "quantized_model.onnx")
            
            # Export model to ONNX
            torch.onnx.export(
                model,
                example_inputs,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            
            # Apply quantization
            if self.config.quantization_type == QuantizationType.DYNAMIC:
                self.quantize_dynamic(onnx_path, quantized_path)
            elif self.config.quantization_type == QuantizationType.STATIC:
                if calibration_dataloader is None:
                    raise ValueError("Static quantization requires calibration data")
                
                # Create calibration data reader
                calibration_data_reader = self._create_calibration_data_reader(calibration_dataloader)
                
                self.quantize_static(
                    onnx_path,
                    quantized_path,
                    calibration_data_reader
                )
            
            logger.info(f"ONNX quantized model saved to {quantized_path}")
            return quantized_path
    
    def _create_calibration_data_reader(self, calibration_dataloader):
        """Create calibration data reader for ONNX static quantization"""
        
        class CalibrationDataReaderImpl(self.CalibrationDataReader):
            def __init__(self, dataloader, num_samples):
                self.dataloader = dataloader
                self.num_samples = num_samples
                self.current_sample = 0
            
            def get_next(self):
                if self.current_sample >= self.num_samples:
                    return None
                
                try:
                    batch = next(iter(self.dataloader))
                    if isinstance(batch, dict):
                        # Take first input
                        input_data = list(batch.values())[0]
                    else:
                        input_data = batch[0] if isinstance(batch, (list, tuple)) else batch
                    
                    self.current_sample += 1
                    return {"input": input_data.numpy()}
                except:
                    return None
        
        return CalibrationDataReaderImpl(calibration_dataloader, self.config.num_calibration_samples)
    
    def get_model_size(self, model_path: str) -> int:
        """Get ONNX model size in bytes"""
        return os.path.getsize(model_path)
    
    def benchmark_model(
        self,
        model_path: str,
        example_inputs: torch.Tensor,
        num_runs: int = 100
    ) -> Dict[str, float]:
        """Benchmark ONNX model performance"""
        import time
        
        # Create ONNX Runtime session
        session = self.onnxruntime.InferenceSession(model_path)
        
        # Prepare inputs
        input_name = session.get_inputs()[0].name
        inputs = {input_name: example_inputs.numpy()}
        
        # Warmup
        for _ in range(10):
            _ = session.run(None, inputs)
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_runs):
            _ = session.run(None, inputs)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        throughput = 1.0 / avg_time
        
        return {
            "avg_inference_time": avg_time,
            "throughput": throughput,
            "total_time": end_time - start_time
        }
    
    def save_quantized_model(self, model_path: str, save_path: str):
        """Save quantized ONNX model"""
        import shutil
        
        os.makedirs(save_path, exist_ok=True)
        shutil.copy(model_path, os.path.join(save_path, "quantized_model.onnx"))
        
        logger.info(f"ONNX quantized model saved to {save_path}")


# Convenience classes for specific quantization types
class DynamicQuantizer(ModelQuantizer):
    """Dynamic quantization convenience class"""
    
    def __init__(self, bits: int = 8, backend: str = "pytorch"):
        config = QuantizationConfig(
            quantization_type=QuantizationType.DYNAMIC,
            backend=QuantizationBackend(backend),
            bits=bits,
            optimize_for_inference=True
        )
        super().__init__(config)


class StaticQuantizer(ModelQuantizer):
    """Static quantization convenience class"""
    
    def __init__(self, bits: int = 8, backend: str = "pytorch", calibration_samples: int = 100):
        config = QuantizationConfig(
            quantization_type=QuantizationType.STATIC,
            backend=QuantizationBackend(backend),
            bits=bits,
            num_calibration_samples=calibration_samples,
            optimize_for_inference=True
        )
        super().__init__(config)
