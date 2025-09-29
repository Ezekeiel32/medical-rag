# MLE-STAR Research Plan: Improving Hebrew Medical RAG System

## Executive Summary

This research plan outlines a comprehensive ML engineering approach to improve the Hebrew medical RAG system's answer relevancy from 0.058 to the target of 0.85. The plan employs Google's STAR methodology (Situation, Task, Action, Result) with an agent-driven research framework.

## Current System Analysis (SITUATION)

### Baseline Performance Metrics
- **Faithfulness**: 0.67 (acceptable)
- **Answer Relevancy**: 0.058 (far below target)
- **Architecture**: Multi-vector FAISS + Ollama/Qwen2.5 + Deterministic extractors

### System Components
- **Retrieval**: FAISS vector search with multi-vector embeddings
- **Generation**: Qwen2.5 model with Hebrew medical prompts
- **Evaluation**: RAGAS framework with Gemini judge
- **Data**: Hebrew OCR medical documents with structured metadata

### Key Challenges Identified
1. **Answer Relevancy Bottleneck**: Judge doesn't recognize generated answers as relevant
2. **Context Synthesis Gap**: Rich content exists but isn't effectively utilized
3. **Hebrew Medical Domain Gap**: Lack of domain-specific optimization
4. **Retrieval Precision Issues**: May be retrieving suboptimal contexts

## Research Objectives (TASK)

### Primary Goal
Achieve RAGAS answer relevancy score of ≥0.85 for Hebrew medical queries

### Secondary Goals
- Improve faithfulness to ≥0.90
- Reduce answer generation latency by 30%
- Enhance context utilization efficiency
- Develop domain-specific evaluation metrics

### Success Criteria
- Answer relevancy ≥0.85 on held-out test set
- Faithfulness ≥0.90 across all question types
- Latency <3 seconds per query
- Human evaluation satisfaction >95%

## Research Methodology (ACTION)

### Phase 1: Foundation & Data (Weeks 1-2)

#### 1.1 Data Augmentation Pipeline
```python
# Generate domain-specific training data
class DataAugmentor:
    def generate_medical_qa_pairs():
        # Medical condition variations
        conditions = ["כאבים", "הגבלה תפקודית", "חזרה לעבודה"]
        # Generate variations using medical terminology
        # Include temporal expressions (נכון ל-, עד תאריך)
        # Add document type variations
        return synthetic_dataset
```

#### 1.2 Quality Validation Framework
```python
# Automated quality checks
quality_checks = {
    'ocr_accuracy': validate_ocr_quality(),
    'date_extraction': validate_date_parsing(),
    'medical_terms': validate_medical_vocabulary(),
    'contextual_coherence': validate_answer_coherence()
}
```

#### 1.3 Baseline Establishment
- Current RAGAS scores: Faithfulness 0.67, Relevancy 0.058
- Performance profiling: latency, memory usage, error patterns
- Failure analysis: categorize low-relevancy patterns

### Phase 2: Retrieval Optimization (Weeks 3-4)

#### 2.1 Multi-Stage Retrieval Architecture
```python
class EnhancedRetriever:
    def retrieve_optimized(query, ocr_dir):
        # Stage 1: Semantic retrieval (top-20)
        candidates = faiss_search(query, top_k=20)
        
        # Stage 2: Query expansion with medical terms
        expanded_query = expand_medical_query(query)
        expanded_candidates = faiss_search(expanded_query, top_k=15)
        
        # Stage 3: Domain-specific re-ranking
        medical_scores = score_medical_relevance(candidates)
        temporal_scores = score_temporal_relevance(candidates)
        
        # Stage 4: Hybrid ranking
        final_scores = combine_scores(
            semantic_scores, medical_scores, temporal_scores
        )
        
        return final_scores[:8]
```

#### 2.2 Embedding Model Optimization
- **Current**: intfloat/multilingual-e5-large
- **Candidates**: 
  - sentence-transformers/paraphrase-multilingual-mpnet-base-v2
  - microsoft/DialoGPT-medium (Hebrew fine-tuned)
  - onlplab/alephbert-base (Hebrew BERT)
- **Evaluation**: Retrieval precision@8, medical term coverage

#### 2.3 Context Compression Strategies
```python
class ContextCompressor:
    def compress_contexts(contexts, max_tokens=800):
        # Extract relevant snippets using medical NLP
        medical_entities = extract_medical_entities(contexts)
        temporal_references = extract_temporal_info(contexts)
        
        # Preserve causal and temporal relationships
        compressed = preserve_relationships(
            contexts, medical_entities, temporal_references
        )
        
        return compressed
```

### Phase 3: Generation Optimization (Weeks 5-6)

#### 3.1 Prompt Engineering Framework
```python
# Systematic prompt optimization
class PromptOptimizer:
    def optimize_medical_prompts():
        # A/B testing of prompt templates
        templates = {
            'clinical': "בהתבסס על התיעוד הרפואי...",
            'temporal': "נכון לתאריך [DATE]...",
            'occupational': "בהתייחס להמלצות הרופא התעסוקתי..."
        }
        
        # LLM-based prompt refinement
        for template in templates:
            refined = refine_with_gpt4(template, examples)
        
        return best_performing_template
```

#### 3.2 Model Fine-tuning Strategy
```python
# Domain-specific fine-tuning
fine_tuning_config = {
    'base_model': 'qwen2.5:7b-instruct',
    'training_data': 'hebrew_medical_qa_pairs.jsonl',
    'learning_rate': 2e-5,
    'epochs': 3,
    'lora_config': {
        'r': 16,
        'lora_alpha': 32,
        'target_modules': ['q_proj', 'k_proj', 'v_proj']
    }
}
```

#### 3.3 Answer Structure Standardization
```python
# Consistent answer format for judge alignment
answer_format = """
## מצבו התפקודי
**[STATUS]** ([DATE])
**מסמך**: [DOC_TYPE]
**ציטוט**: "[SNIPPET]"
**ניתוח**: [ANALYSIS]
"""
```

### Phase 4: Evaluation & Iteration (Weeks 7-8)

#### 4.1 Automated Evaluation Agent
```python
class RAGASEvaluationAgent:
    def run_evaluation_cycle():
        # Build evaluation dataset
        dataset = self.build_evaluation_set()
        
        # Run comprehensive evaluation
        scores = self.evaluate_comprehensive(dataset)
        
        # Analyze failures
        failures = self.analyze_failure_patterns(scores)
        
        # Generate improvement hypotheses
        hypotheses = self.generate_improvements(failures)
        
        # Implement changes
        self.implement_changes(hypotheses)
        
        return scores, hypotheses
```

#### 4.2 Failure Analysis Pipeline
- **Pattern Recognition**: Categorize low-relevancy failures
- **Root Cause Analysis**: Identify retrieval vs generation issues
- **Correlation Analysis**: Link context features to performance

#### 4.3 Bayesian Optimization Framework
```python
# Automated hyperparameter tuning
optimization_space = {
    'retrieval_top_k': (3, 15),
    'similarity_threshold': (0.1, 0.9),
    'context_window': (200, 800),
    'rerank_weight': (0.1, 0.9),
    'temperature': (0.1, 0.8),
    'max_tokens': (100, 500)
}

optimizer = BayesianOptimizer(
    space=optimization_space,
    objective=evaluate_ragas_relevancy,
    max_evals=50
)
```

## Technical Implementation Details

### Infrastructure Requirements
- **Compute Resources**: GPU cluster (A100/H100) for fine-tuning
- **Storage**: 500GB for datasets, models, and experiments
- **Monitoring**: MLflow for experiment tracking
- **CI/CD**: Automated testing and deployment pipelines

### Key Technologies
- **Core Framework**: Google ADK for agent orchestration
- **Models**: Qwen2.5, Gemini, multilingual embeddings
- **Evaluation**: RAGAS, custom Hebrew medical metrics
- **Infrastructure**: FAISS, LangChain, HuggingFace

### Risk Mitigation Strategy
- **Model Safety**: Medical content validation and bias checks
- **Data Privacy**: HIPAA-compliant processing
- **Performance**: Automated rollback mechanisms
- **Cost Control**: Resource usage monitoring and optimization

## Timeline & Milestones

### Week-by-Week Breakdown
- **Week 1**: Data collection and baseline evaluation
- **Week 2**: Synthetic data generation and quality validation
- **Week 3**: Retrieval optimization and embedding experiments
- **Week 4**: Multi-stage retrieval implementation and testing
- **Week 5**: Prompt engineering and generation optimization
- **Week 6**: Model fine-tuning and answer format standardization
- **Week 7**: Automated evaluation agent development
- **Week 8**: Iterative optimization and final evaluation
- **Week 9**: Production deployment preparation
- **Week 10**: A/B testing and performance monitoring

### Success Milestones
- **Week 2**: Synthetic dataset with 95% quality score
- **Week 4**: Retrieval precision@8 >80%
- **Week 6**: Generation quality improvement >50%
- **Week 8**: Answer relevancy >0.70
- **Week 10**: Target 0.85 achieved and deployed

## Success Metrics & KPIs

### Primary Metrics
- **Answer Relevancy**: RAGAS score ≥0.85
- **Faithfulness**: RAGAS score ≥0.90
- **Latency**: <3 seconds per query
- **Throughput**: >100 queries/minute

### Secondary Metrics
- **Medical Accuracy**: Domain expert validation >95%
- **User Satisfaction**: A/B testing preference >90%
- **System Reliability**: 99.9% uptime
- **Cost Efficiency**: <$0.01 per query

### Evaluation Framework
```python
comprehensive_evaluation = {
    'ragas_metrics': ['faithfulness', 'answer_relevancy', 'context_relevancy'],
    'domain_metrics': ['medical_accuracy', 'temporal_accuracy', 'completeness'],
    'performance_metrics': ['latency', 'throughput', 'memory_usage'],
    'user_metrics': ['satisfaction_score', 'preference_rate']
}
```

## Resource Requirements

### Team Composition
- **MLE Lead**: 1 FTE (research planning and execution)
- **ML Engineer**: 1 FTE (implementation and optimization)
- **Medical Domain Expert**: 0.5 FTE (validation and guidance)
- **DevOps Engineer**: 0.5 FTE (infrastructure and deployment)

### Budget Allocation
- **Compute Costs**: $5,000/month (GPU instances)
- **API Costs**: $2,000/month (Gemini evaluation calls)
- **Storage**: $500/month (cloud storage)
- **Tools**: $1,000/month (MLflow, monitoring)

## Risk Assessment & Contingency Plans

### Technical Risks
- **Model Performance Degradation**: Implement automated rollback
- **Data Quality Issues**: Additional validation and cleaning
- **Integration Challenges**: Modular design with fallback options
- **Scalability Concerns**: Performance profiling and optimization

### Timeline Risks
- **Scope Creep**: Fixed milestone reviews and scope control
- **Technical Blockers**: Parallel experimentation tracks
- **Resource Constraints**: Prioritized task allocation
- **External Dependencies**: Backup solutions and alternatives

### Business Risks
- **Budget Overrun**: Cost monitoring and quarterly reviews
- **Timeline Delays**: Phase-gate approvals and progress tracking
- **Quality Concerns**: Comprehensive testing and validation

## Conclusion

This MLE-STAR research plan provides a structured, systematic approach to achieving the 0.85 RAGAS relevancy target for the Hebrew medical RAG system. The plan combines:

- **Rigorous Methodology**: STAR framework with agent-driven research
- **Comprehensive Coverage**: From data to deployment
- **Risk Mitigation**: Multiple fallback strategies
- **Measurable Success**: Clear KPIs and milestones

The research approach ensures that each component is systematically optimized while maintaining the overall system integrity. The agent-driven framework allows for continuous learning and adaptation throughout the research process.

**Expected Outcome**: A production-ready RAG system that consistently achieves ≥0.85 answer relevancy for Hebrew medical queries, with comprehensive evaluation and monitoring capabilities.

---

*This research plan follows Google's MLE-STAR methodology and incorporates best practices from production ML systems. All experiments will be tracked in MLflow, and results will be validated through both automated metrics and human expert review.*
