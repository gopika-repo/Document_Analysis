# 🚀 Vision-Fusion: Multi-Modal Document Intelligence System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/docker-available-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**A production-ready AI agent system that combines Computer Vision and Language Models for intelligent document understanding.**

## 📋 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Quick Start](#-quick-start)
- [Usage Examples](#-usage-examples)
- [API Documentation](#-api-documentation)
- [Testing](#-testing)
- [Deployment](#-deployment)
- [Performance Benchmarks](#-performance-benchmarks)
- [Contributing](#-contributing)
- [License](#-license)

## 🎯 Overview

Vision-Fusion is a cutting-edge multi-modal document intelligence system that processes scanned documents (PDFs, images) by combining:

- **Computer Vision**: YOLOv8 object detection for tables, charts, signatures
- **OCR**: Hybrid Tesseract + PaddleOCR with confidence-based switching
- **LLM Integration**: Grok API with Ollama fallback for semantic analysis
- **Multi-Agent System**: 4 specialized agents orchestrated via LangGraph
- **Multi-Modal RAG**: Qdrant vector database for cross-modal retrieval

## ✨ Features

### 🖼️ **Computer Vision**
- ✅ YOLOv8 object detection (tables, charts, diagrams, signatures)
- ✅ Layout analysis using OpenCV
- ✅ Visual feature extraction
- ✅ Document element classification

### 🔤 **OCR & Text Processing**
- ✅ Hybrid OCR (Tesseract primary, PaddleOCR fallback)
- ✅ Word-level bounding boxes with confidence scores
- ✅ Entity extraction (dates, amounts, names, organizations)
- ✅ Semantic analysis using LLMs

### 🤖 **Multi-Agent System**
| Agent | Responsibility | Technology |
|-------|---------------|------------|
| **Vision Agent** | Visual element detection | YOLOv8 + OpenCV |
| **Text Agent** | OCR & semantic analysis | Tesseract + LLM |
| **Fusion Agent** | Multi-modal integration | LangGraph |
| **Validation Agent** | Quality assurance | Rule-based + ML |

### 🔍 **Multi-Modal RAG**
- ✅ Text embeddings (all-MiniLM-L6-v2)
- ✅ Visual embeddings (CLIP-based features)
- ✅ Cross-modal retrieval
- ✅ Qdrant vector database

### 📊 **Confidence & Validation**
- ✅ Per-field confidence scoring
- ✅ Multi-modal validation rules
- ✅ Contradiction detection
- ✅ Human review workflow

## 🏗️ System Architecture

![Architecture Diagram](architecture.png)

### **Core Components**
Document Ingestion Layer
├── PDF/Image upload
├── Preprocessing (300 DPI conversion)
└── Metadata extraction

Computer Vision Pipeline
├── YOLOv8 object detection
├── Layout analysis
└── Visual feature extraction

OCR Pipeline
├── Tesseract (primary)
├── PaddleOCR (fallback)
└── Confidence-based switching

Multi-Agent System
├── Vision Agent
├── Text Agent
├── Fusion Agent
└── Validation Agent

RAG System
├── Embedding generation
├── Vector storage (Qdrant)
└── Cross-modal retrieval

API Layer
├── FastAPI endpoints
├── Async processing
└── WebSocket support

## 🚀 Quick Start

### **Prerequisites**
- Python 3.10+
- Docker & Docker Compose
- Tesseract OCR (system install)

### **Option 1: Docker (Recommended)**
```bash
# Clone repository
git clone https://github.com/yourusername/vision-fusion.git
cd vision-fusion/backend

# Start all services
docker-compose up --build

# The system will be available at:
# API: http://localhost:8000
# Qdrant Dashboard: http://localhost:6333/dashboard
# API Documentation: http://localhost:8000/docs