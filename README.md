# Pragmatic Drift: A Metric for LLM Reasoning Stability

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This repository contains the official implementation for the research project:

> **"Pragmatic Drift and Context Boundaries: Evaluating 'Translation in the Wild' Within High-Dimensional RAG Architectures"**

## 📄 Overview

Large Language Models (LLMs) frequently hallucinate even when given the correct retrieved documents. This project formalizes **Pragmatic Drift (δp)**—a metric that measures how much an LLM's reasoning shifts when its system instructions are perturbed. High δp indicates fragile reasoning and predicts hallucinations.

I also introduce the **Perturbation-to-Attention (P2A) stress-test**, a deployable guardrail that intercepts unstable contexts before they reach the end-user.

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- A GPU (recommended) or CPU

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/pragmatic-drift.git
   cd pragmatic-drift
