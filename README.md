# RWKVInside

## Overview

RWKVInside is a research repository focused on deep investigation of attention mechanism conversion from traditional transformer architectures to RWKV-based systems. This project implements advanced techniques for transforming standard attention mechanisms while maintaining model performance and efficiency.

## Architecture: cxa079

This repository features the **cxa079** architecture, specifically designed for attention conversion models. The architecture is built upon RWKV-7 foundation with several key optimizations and enhancements.

## Key Features

### Core Architecture
- **RWKV-7 Base**: Utilizes RWKV-7 as the foundational architecture for attention conversion
- **GroupNorm Removal**: Streamlined architecture with GroupNorm layers removed for improved efficiency
- **Key Residual (K_FIRST)**: Enhanced with Key Residual connections to improve information flow

### Attention Mechanisms
- **Group Query Style**: Supports both Group Query Attention and Multi-Head Attention (MHA) configurations
- **Attention Freeze Support**: Enables selective freezing of attention parameters during training

### Optimization Features
- **LoRA Size Optimization**: Optimized Low-Rank Adaptation (LoRA) parameter sizing for efficient fine-tuning
- **Bitsandbytes Quantization**: Integrated quantization support for memory-efficient deployment
- **LoRA & Bone Compatibility**: Full support for both LoRA and Bone adapter methodologies

### Training Capabilities
- **Hybrid Simultaneous Learning**: Advanced training methodology supporting concurrent optimization of multiple model components
- **Parameter-Specific Control**: Fine-grained control over parameter learning and freezing strategies

## Technical Specifications

| Component | Implementation | Status |
|-----------|---------------|---------|
| Base Architecture | RWKV-7 | ✓ Implemented |
| Normalization | GroupNorm Removed | ✓ Implemented |
| Residual Connections | K_FIRST Key Residual | ✓ Implemented |
| Attention Style | Group Query / MHA | ✓ Implemented |
| Adapter Support | LoRA Optimized | ✓ Implemented |
| Quantization | Bitsandbytes | ✓ Implemented |
| Training Mode | Hybrid Simultaneous | ✓ Implemented |
| Parameter Control | Attention Freeze | ✓ Implemented |

 

## Research Objectives

This project aims to:

1. **Investigate attention conversion methodologies** from transformer-based architectures to RWKV systems
2. **Optimize parameter efficiency** through advanced adapter techniques and quantization
3. **Develop hybrid training approaches** for simultaneous optimization of multiple model components
4. **Evaluate performance characteristics** of converted attention mechanisms

## Performance Considerations

The cxa079 architecture incorporates several optimizations:

- **Memory Efficiency**: Reduced memory footprint through GroupNorm removal and quantization
- **Training Stability**: Enhanced gradient flow via Key Residual connections
- **Computational Efficiency**: Optimized attention computation through Group Query mechanisms
- **Flexibility**: Support for multiple adapter types and training configurations

 

## Contributing

This repository serves as a personal research project. Contributions, suggestions, and discussions regarding attention conversion methodologies are welcome through issues and pull requests.

## License

Apache 2.0

 

## Contact

For technical questions or research collaboration inquiries, please open an issue in this repository.