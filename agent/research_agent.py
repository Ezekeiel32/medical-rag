#!/usr/bin/env python3
"""
RAG Research Agent - Systematic Investigation of RAG System Improvements

This agent orchestrates research into various aspects of the RAG system:
- Embedding model alternatives
- Retrieval strategy improvements  
- Generation model optimization
- Evaluation framework enhancements
"""

import json
import os
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import subprocess

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a RAG experiment."""
    name: str
    description: str
    embedding_model: Optional[str] = None
    generation_model: Optional[str] = None
    chunking_params: Optional[Dict[str, Any]] = None
    retrieval_params: Optional[Dict[str, Any]] = None
    evaluation_params: Optional[Dict[str, Any]] = None


@dataclass
class ExperimentResult:
    """Results from a RAG experiment."""
    config: ExperimentConfig
    metrics: Dict[str, float]
    execution_time: float
    timestamp: str
    notes: str = ""


class RAGResearchAgent:
    """Main research agent for RAG system optimization."""
    
    def __init__(self, base_ocr_dir: str = "/home/chezy/rag_medical/ocr_out"):
        self.base_ocr_dir = base_ocr_dir
        self.results_dir = "research_results"
        self.experiments_log = "experiments.jsonl"
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Research phases
        self.phases = [
            "baseline_establishment",
            "embedding_research", 
            "retrieval_research",
            "generation_research",
            "integration_optimization"
        ]
        
        self.current_phase = "baseline_establishment"
        
        # Experiment configurations to test
        self.embedding_models_to_test = [
            "intfloat/multilingual-e5-large",  # Current baseline
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            "intfloat/multilingual-e5-base", 
            "sentence-transformers/LaBSE",
            "sentence-transformers/distiluse-base-multilingual-cased",
        ]
        
        self.generation_models_to_test = [
            "qwen2.5:7b-instruct",  # Current baseline
            "mistral:7b-instruct",
            "gemma2:7b-instruct", 
            "llama3.2:7b-instruct",
        ]
    
    def log_experiment(self, result: ExperimentResult) -> None:
        """Log experiment results to JSONL file."""
        with open(self.experiments_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
    
    def run_baseline_evaluation(self) -> Dict[str, float]:
        """Run baseline evaluation using current system configuration."""
        logger.info("Running baseline evaluation...")
        
        start_time = time.time()
        
        # Run current RAGAS evaluation
        try:
            # Build RAGAS dataset
            subprocess.run([
                "python", "-m", "src.cli", "ocr-build-ragas",
                "--ocr-dir", self.base_ocr_dir,
                "--model", "qwen2.5:7b-instruct",
                "--top-k", "6"
            ], check=True, cwd="/home/chezy/rag_medical")
            
            # Run RAGAS evaluation  
            ragas_file = os.path.join(self.base_ocr_dir, "ragas_he.jsonl")
            result = subprocess.run([
                "python", "-m", "src.cli", "ragas-eval",
                "--input", ragas_file
            ], capture_output=True, text=True, cwd="/home/chezy/rag_medical")
            
            if result.returncode == 0:
                metrics = json.loads(result.stdout)
                execution_time = time.time() - start_time
                
                logger.info(f"Baseline metrics: {metrics}")
                return {
                    **metrics,
                    "execution_time": execution_time
                }
            else:
                logger.error(f"RAGAS evaluation failed: {result.stderr}")
                return {"error": "evaluation_failed", "execution_time": time.time() - start_time}
                
        except Exception as e:
            logger.error(f"Baseline evaluation error: {e}")
            return {"error": str(e), "execution_time": time.time() - start_time}
    
    def test_embedding_model(self, model_name: str, top_k: int = 6) -> Dict[str, float]:
        """Test a specific embedding model."""
        logger.info(f"Testing embedding model: {model_name}")
        
        start_time = time.time()
        
        try:
            # Re-index with new embedding model
            subprocess.run([
                "python", "-m", "src.cli", "ocr-index",
                "--ocr-dir", self.base_ocr_dir,
                "--model", model_name,
                "--chunk-chars", "900",
                "--overlap", "120"
            ], check=True, cwd="/home/chezy/rag_medical")
            
            # Build RAGAS dataset with new index
            subprocess.run([
                "python", "-m", "src.cli", "ocr-build-ragas", 
                "--ocr-dir", self.base_ocr_dir,
                "--model", "qwen2.5:7b-instruct",
                "--top-k", str(top_k)
            ], check=True, cwd="/home/chezy/rag_medical")
            
            # Run RAGAS evaluation
            ragas_file = os.path.join(self.base_ocr_dir, "ragas_he.jsonl")
            result = subprocess.run([
                "python", "-m", "src.cli", "ragas-eval",
                "--input", ragas_file
            ], capture_output=True, text=True, cwd="/home/chezy/rag_medical")
            
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                metrics = json.loads(result.stdout)
                return {
                    **metrics,
                    "execution_time": execution_time,
                    "model": model_name
                }
            else:
                logger.error(f"Evaluation failed for {model_name}: {result.stderr}")
                return {
                    "error": "evaluation_failed", 
                    "execution_time": execution_time,
                    "model": model_name
                }
                
        except Exception as e:
            logger.error(f"Error testing {model_name}: {e}")
            return {
                "error": str(e), 
                "execution_time": time.time() - start_time,
                "model": model_name
            }
    
    def test_generation_model(self, model_name: str, top_k: int = 6) -> Dict[str, float]:
        """Test a specific generation model."""
        logger.info(f"Testing generation model: {model_name}")
        
        start_time = time.time()
        
        try:
            # Build RAGAS dataset with new generation model
            subprocess.run([
                "python", "-m", "src.cli", "ocr-build-ragas",
                "--ocr-dir", self.base_ocr_dir, 
                "--model", model_name,
                "--top-k", str(top_k)
            ], check=True, cwd="/home/chezy/rag_medical")
            
            # Run RAGAS evaluation
            ragas_file = os.path.join(self.base_ocr_dir, "ragas_he.jsonl")
            result = subprocess.run([
                "python", "-m", "src.cli", "ragas-eval",
                "--input", ragas_file
            ], capture_output=True, text=True, cwd="/home/chezy/rag_medical")
            
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                metrics = json.loads(result.stdout)
                return {
                    **metrics,
                    "execution_time": execution_time,
                    "model": model_name
                }
            else:
                logger.error(f"Evaluation failed for {model_name}: {result.stderr}")
                return {
                    "error": "evaluation_failed",
                    "execution_time": execution_time, 
                    "model": model_name
                }
                
        except Exception as e:
            logger.error(f"Error testing {model_name}: {e}")
            return {
                "error": str(e),
                "execution_time": time.time() - start_time,
                "model": model_name
            }
    
    def run_embedding_research_phase(self) -> List[ExperimentResult]:
        """Run systematic research on embedding models."""
        logger.info("Starting embedding model research phase...")
        
        results = []
        
        for model in self.embedding_models_to_test:
            config = ExperimentConfig(
                name=f"embedding_test_{model.replace('/', '_')}",
                description=f"Testing embedding model: {model}",
                embedding_model=model
            )
            
            metrics = self.test_embedding_model(model)
            
            result = ExperimentResult(
                config=config,
                metrics=metrics,
                execution_time=metrics.get("execution_time", 0),
                timestamp=datetime.now().isoformat(),
                notes=f"Embedding model comparison: {model}"
            )
            
            results.append(result)
            self.log_experiment(result)
            
            logger.info(f"Completed test for {model}")
        
        return results
    
    def run_generation_research_phase(self) -> List[ExperimentResult]:
        """Run systematic research on generation models."""
        logger.info("Starting generation model research phase...")
        
        results = []
        
        for model in self.generation_models_to_test:
            config = ExperimentConfig(
                name=f"generation_test_{model.replace(':', '_')}",
                description=f"Testing generation model: {model}",
                generation_model=model
            )
            
            metrics = self.test_generation_model(model)
            
            result = ExperimentResult(
                config=config,
                metrics=metrics, 
                execution_time=metrics.get("execution_time", 0),
                timestamp=datetime.now().isoformat(),
                notes=f"Generation model comparison: {model}"
            )
            
            results.append(result)
            self.log_experiment(result)
            
            logger.info(f"Completed test for {model}")
        
        return results
    
    def generate_research_report(self) -> str:
        """Generate comprehensive research report."""
        logger.info("Generating research report...")
        
        # Load all experiment results
        experiments = []
        if os.path.exists(self.experiments_log):
            with open(self.experiments_log, "r", encoding="utf-8") as f:
                for line in f:
                    experiments.append(json.loads(line))
        
        # Analyze results
        report_lines = [
            "# RAG System Research Report",
            f"Generated: {datetime.now().isoformat()}",
            f"Total experiments: {len(experiments)}",
            "",
            "## Executive Summary",
            "",
        ]
        
        # Find best performing configurations
        if experiments:
            # Group by experiment type
            embedding_exps = [e for e in experiments if e['config']['embedding_model']]
            generation_exps = [e for e in experiments if e['config']['generation_model']]
            
            # Best embedding model
            if embedding_exps:
                best_embedding = max(embedding_exps, 
                                   key=lambda x: x['metrics'].get('faithfulness', 0))
                report_lines.extend([
                    f"### Best Embedding Model: {best_embedding['config']['embedding_model']}",
                    f"- Faithfulness: {best_embedding['metrics'].get('faithfulness', 'N/A')}",
                    f"- Answer Relevancy: {best_embedding['metrics'].get('answer_relevancy', 'N/A')}",
                    f"- Context Precision: {best_embedding['metrics'].get('context_precision', 'N/A')}",
                    ""
                ])
            
            # Best generation model  
            if generation_exps:
                best_generation = max(generation_exps,
                                    key=lambda x: x['metrics'].get('faithfulness', 0))
                report_lines.extend([
                    f"### Best Generation Model: {best_generation['config']['generation_model']}",
                    f"- Faithfulness: {best_generation['metrics'].get('faithfulness', 'N/A')}",
                    f"- Answer Relevancy: {best_generation['metrics'].get('answer_relevancy', 'N/A')}",
                    f"- Context Precision: {best_generation['metrics'].get('context_precision', 'N/A')}",
                    ""
                ])
        
        report_lines.extend([
            "## Detailed Results",
            "",
            "### All Experiments",
            ""
        ])
        
        # Add detailed results table
        for i, exp in enumerate(experiments, 1):
            config = exp['config']
            metrics = exp['metrics']
            report_lines.extend([
                f"#### Experiment {i}: {config['name']}",
                f"- Description: {config['description']}",
                f"- Timestamp: {exp['timestamp']}",
                f"- Execution Time: {exp['execution_time']:.2f}s",
                f"- Metrics: {json.dumps(metrics, ensure_ascii=False, indent=2)}",
                ""
            ])
        
        report_content = "\n".join(report_lines)
        
        # Save report
        report_path = os.path.join(self.results_dir, "research_report.md") 
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        
        logger.info(f"Research report saved to: {report_path}")
        return report_path
    
    def run_full_research_session(self) -> str:
        """Run complete research session."""
        logger.info("Starting full RAG research session...")
        
        session_start = time.time()
        
        # Phase 1: Baseline
        logger.info("Phase 1: Establishing baseline...")
        baseline_metrics = self.run_baseline_evaluation()
        
        baseline_result = ExperimentResult(
            config=ExperimentConfig(
                name="baseline",
                description="Current system baseline",
                embedding_model="intfloat/multilingual-e5-large",
                generation_model="qwen2.5:7b-instruct"
            ),
            metrics=baseline_metrics,
            execution_time=baseline_metrics.get("execution_time", 0),
            timestamp=datetime.now().isoformat(),
            notes="Baseline measurement of current system"
        )
        
        self.log_experiment(baseline_result)
        
        # Phase 2: Embedding research
        logger.info("Phase 2: Embedding model research...")
        embedding_results = self.run_embedding_research_phase()
        
        # Phase 3: Generation research  
        logger.info("Phase 3: Generation model research...")
        generation_results = self.run_generation_research_phase()
        
        # Generate report
        logger.info("Generating final research report...")
        report_path = self.generate_research_report()
        
        total_time = time.time() - session_start
        logger.info(f"Research session completed in {total_time:.2f} seconds")
        
        return report_path


def main():
    """Main entry point for research agent."""
    agent = RAGResearchAgent()
    
    try:
        report_path = agent.run_full_research_session()
        print(f"Research completed. Report available at: {report_path}")
    except KeyboardInterrupt:
        print("\nResearch session interrupted by user.")
    except Exception as e:
        logger.error(f"Research session failed: {e}")
        raise


if __name__ == "__main__":
    main()

