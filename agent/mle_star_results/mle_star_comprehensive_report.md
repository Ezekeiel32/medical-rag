# MLE-STAR Research Report: Hebrew Medical RAG Optimization
Generated: 2025-09-06T12:49:57.034883

## Executive Summary

This report documents the systematic MLE-STAR research approach to improve
Hebrew medical RAG system answer relevancy from 0.058 to the target ≥0.85.

## Current Situation Analysis

### Baseline Performance
- **Faithfulness**: nan
- **Answer Relevancy**: nan
- **Target Faithfulness**: 0.900
- **Target Relevancy**: 0.850

## Foundation & Data Results

Completed 1 experiments:

### data_augmentation
- **Configuration**: {
  "configs_tested": [
    {
      "model": "qwen2.5:7b-instruct",
      "top_k": 6,
      "name": "baseline"
    },
    {
      "model": "qwen2.5:7b-instruct",
      "top_k": 10,
      "name": "expanded"
    },
    {
      "model": "mistral:7b-instruct",
      "top_k": 6,
      "name": "mistral_baseline"
    }
  ]
}
- **Metrics**: {
  "datasets_created": 3,
  "total_samples": 9
}
- **Duration**: 88.88s

## Retrieval Optimization Results

Completed 4 experiments:

### embedding_test_intfloat_multilingual-e5-large
- **Configuration**: {
  "embedding_model": "intfloat/multilingual-e5-large",
  "chunk_chars": 900,
  "overlap_chars": 120
}
- **Metrics**: {
  "faithfulness": 0.6333333333333333,
  "answer_relevancy": 0.05194870593874262
}
- **Duration**: 204.56s

### embedding_test_sentence-transformers_paraphrase-multilingual-mpnet-base-v2
- **Configuration**: {
  "embedding_model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
  "chunk_chars": 900,
  "overlap_chars": 120
}
- **Metrics**: {
  "faithfulness": 0.6666666666666666,
  "answer_relevancy": 0.023627047281835383
}
- **Duration**: 169.67s

### embedding_test_intfloat_multilingual-e5-base
- **Configuration**: {
  "embedding_model": "intfloat/multilingual-e5-base",
  "chunk_chars": 900,
  "overlap_chars": 120
}
- **Metrics**: {
  "faithfulness": 0.6333333333333333,
  "answer_relevancy": 0.0437904617548047
}
- **Duration**: 139.51s

### embedding_test_sentence-transformers_LaBSE
- **Configuration**: {
  "embedding_model": "sentence-transformers/LaBSE",
  "chunk_chars": 900,
  "overlap_chars": 120
}
- **Metrics**: {
  "faithfulness": 0.5925925925925926,
  "answer_relevancy": 0.040721179024831224
}
- **Duration**: 302.68s

## Generation Optimization Results

Completed 3 experiments:

### generation_test_qwen2.5_7b-instruct
- **Configuration**: {
  "generation_model": "qwen2.5:7b-instruct",
  "top_k": 6
}
- **Metrics**: {
  "faithfulness": 0.6666666666666666,
  "answer_relevancy": 0.043807902000154365
}
- **Duration**: 52.31s

### generation_test_mistral_7b-instruct
- **Configuration**: {
  "generation_model": "mistral:7b-instruct",
  "top_k": 6
}
- **Metrics**: {
  "faithfulness": 0.5,
  "answer_relevancy": 0.04558254942682786
}
- **Duration**: 51.65s

### generation_test_gemma2_7b-instruct
- **Configuration**: {
  "generation_model": "gemma2:7b-instruct",
  "top_k": 6
}
- **Metrics**: {
  "faithfulness": 0.625,
  "answer_relevancy": 0.04367722248608879
}
- **Duration**: 48.76s
