#!/usr/bin/env python3
"""
Google ADK Research Orchestrator for RAG System Optimization

This module uses Google's Agent Development Kit to orchestrate sophisticated
research sessions for RAG system improvement, with advanced analysis and 
recommendation capabilities.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

# Import Google ADK components (will work once API key is configured)
try:
    from google.adk.agents import Agent, LlmAgent
    from google.adk.tools import agent_tool
    ADK_AVAILABLE = True
except ImportError:
    print("Google ADK not available. Please ensure API key is configured.")
    ADK_AVAILABLE = False

from research_agent import RAGResearchAgent, ExperimentResult


class ADKResearchOrchestrator:
    """Advanced research orchestrator using Google ADK."""
    
    def __init__(self, base_ocr_dir: str = "/home/chezy/rag_medical/ocr_out"):
        self.base_ocr_dir = base_ocr_dir
        self.core_agent = RAGResearchAgent(base_ocr_dir)
        
        if ADK_AVAILABLE:
            self._setup_adk_agents()
        else:
            print("ADK agents not available. Running in fallback mode.")
    
    def _setup_adk_agents(self):
        """Set up specialized ADK agents for different research tasks."""
        
        # Embedding Research Agent
        self.embedding_researcher = LlmAgent(
            name="EmbeddingResearcher",
            model="gemini-2.0-flash",
            description="Specialized agent for analyzing embedding model performance in Hebrew medical RAG systems.",
            instruction="""You are an expert in embedding models for multilingual medical text retrieval.
            Analyze embedding model performance data and provide insights on:
            1. Model suitability for Hebrew medical text
            2. Performance trade-offs (accuracy vs speed)
            3. Recommendations for optimal configurations
            4. Potential issues with domain adaptation
            
            Always provide specific, actionable recommendations based on the metrics."""
        )
        
        # Generation Research Agent
        self.generation_researcher = LlmAgent(
            name="GenerationResearcher", 
            model="gemini-2.0-flash",
            description="Specialized agent for analyzing generation model performance in Hebrew medical RAG systems.",
            instruction="""You are an expert in large language models for Hebrew medical text generation.
            Analyze generation model performance and provide insights on:
            1. Hebrew language generation quality
            2. Medical domain accuracy and safety
            3. Hallucination detection and mitigation
            4. Response consistency and reliability
            
            Focus on practical improvements for medical RAG applications."""
        )
        
        # Retrieval Strategy Agent
        self.retrieval_strategist = LlmAgent(
            name="RetrievalStrategist",
            model="gemini-2.0-flash", 
            description="Expert in retrieval strategies for medical document search systems.",
            instruction="""You are a retrieval system expert specializing in medical document search.
            Analyze retrieval performance and recommend improvements for:
            1. Semantic search optimization
            2. Hybrid retrieval strategies (dense + sparse)
            3. Query expansion and transformation
            4. Domain-specific filtering and ranking
            
            Provide concrete implementation suggestions with expected impact."""
        )
        
        # Performance Analyst Agent
        self.performance_analyst = LlmAgent(
            name="PerformanceAnalyst",
            model="gemini-2.0-flash",
            description="Performance analysis expert for RAG system optimization.",
            instruction="""You are a performance optimization expert for RAG systems.
            Analyze system performance metrics and identify:
            1. Bottlenecks and inefficiencies
            2. Scalability concerns
            3. Resource utilization optimization
            4. Latency and throughput improvements
            
            Provide prioritized optimization recommendations with implementation complexity estimates."""
        )
        
        # Master Research Coordinator
        self.master_coordinator = LlmAgent(
            name="MasterCoordinator",
            model="gemini-2.0-flash",
            description="Master research coordinator that synthesizes insights from all specialized agents.",
            instruction="""You are the master research coordinator overseeing RAG system optimization.
            Synthesize insights from specialized agents and create:
            1. Comprehensive improvement roadmap
            2. Priority rankings for different optimizations
            3. Resource allocation recommendations
            4. Risk assessment for proposed changes
            
            Provide executive-level recommendations with clear business value.""",
            tools=[
                agent_tool.AgentTool(agent=self.embedding_researcher),
                agent_tool.AgentTool(agent=self.generation_researcher),
                agent_tool.AgentTool(agent=self.retrieval_strategist), 
                agent_tool.AgentTool(agent=self.performance_analyst)
            ]
        )
    
    def analyze_embedding_results(self, results: List[ExperimentResult]) -> Dict[str, Any]:
        """Analyze embedding model experiment results using ADK."""
        if not ADK_AVAILABLE:
            return self._fallback_embedding_analysis(results)
        
        # Prepare data for analysis
        results_data = []
        for result in results:
            results_data.append({
                "model": result.config.embedding_model or "unknown",
                "metrics": result.metrics,
                "execution_time": result.execution_time,
                "notes": result.notes
            })
        
        # Use ADK agent for analysis
        query = f"""
        Analyze these embedding model experiment results for Hebrew medical RAG:
        
        Results: {json.dumps(results_data, ensure_ascii=False, indent=2)}
        
        Provide:
        1. Best performing model with justification
        2. Key performance insights
        3. Trade-off analysis (accuracy vs speed vs memory)
        4. Specific recommendations for optimization
        """
        
        try:
            analysis = self.embedding_researcher.execute(query)
            return {
                "analysis": analysis,
                "best_model": self._extract_best_model(results, "faithfulness"),
                "raw_results": results_data
            }
        except Exception as e:
            logging.error(f"ADK analysis failed: {e}")
            return self._fallback_embedding_analysis(results)
    
    def analyze_generation_results(self, results: List[ExperimentResult]) -> Dict[str, Any]:
        """Analyze generation model experiment results using ADK."""
        if not ADK_AVAILABLE:
            return self._fallback_generation_analysis(results)
        
        results_data = []
        for result in results:
            results_data.append({
                "model": result.config.generation_model or "unknown", 
                "metrics": result.metrics,
                "execution_time": result.execution_time,
                "notes": result.notes
            })
        
        query = f"""
        Analyze these generation model experiment results for Hebrew medical RAG:
        
        Results: {json.dumps(results_data, ensure_ascii=False, indent=2)}
        
        Provide:
        1. Best performing model for medical accuracy
        2. Hebrew language quality assessment
        3. Safety and hallucination analysis
        4. Performance vs quality trade-offs
        5. Specific configuration recommendations
        """
        
        try:
            analysis = self.generation_researcher.execute(query)
            return {
                "analysis": analysis,
                "best_model": self._extract_best_model(results, "faithfulness"),
                "raw_results": results_data
            }
        except Exception as e:
            logging.error(f"ADK analysis failed: {e}")
            return self._fallback_generation_analysis(results)
    
    def generate_master_recommendations(self, all_results: List[ExperimentResult]) -> Dict[str, Any]:
        """Generate master recommendations using the coordinating agent."""
        if not ADK_AVAILABLE:
            return self._fallback_master_recommendations(all_results)
        
        # Group results by type
        embedding_results = [r for r in all_results if r.config.embedding_model]
        generation_results = [r for r in all_results if r.config.generation_model]
        
        # Get individual analyses
        embedding_analysis = self.analyze_embedding_results(embedding_results)
        generation_analysis = self.analyze_generation_results(generation_results)
        
        # Master coordination query
        query = f"""
        As the master research coordinator, synthesize these research findings:
        
        Embedding Analysis:
        {embedding_analysis.get('analysis', 'No analysis available')}
        
        Generation Analysis: 
        {generation_analysis.get('analysis', 'No analysis available')}
        
        Provide:
        1. Executive summary of key findings
        2. Prioritized improvement roadmap
        3. Resource allocation recommendations
        4. Risk assessment and mitigation strategies
        5. Expected impact and ROI for each recommendation
        6. Implementation timeline suggestions
        
        Focus on actionable, high-impact improvements for the Hebrew medical RAG system.
        """
        
        try:
            master_analysis = self.master_coordinator.execute(query)
            return {
                "master_recommendations": master_analysis,
                "embedding_analysis": embedding_analysis,
                "generation_analysis": generation_analysis,
                "summary": self._create_executive_summary(all_results)
            }
        except Exception as e:
            logging.error(f"Master coordination failed: {e}")
            return self._fallback_master_recommendations(all_results)
    
    def run_advanced_research_session(self) -> str:
        """Run advanced research session with ADK orchestration."""
        logging.info("Starting advanced RAG research session with ADK orchestration...")
        
        # Run core research using the base agent
        logging.info("Phase 1: Core research execution...")
        baseline_metrics = self.core_agent.run_baseline_evaluation()
        embedding_results = self.core_agent.run_embedding_research_phase()
        generation_results = self.core_agent.run_generation_research_phase()
        
        # Combine all results
        all_results = embedding_results + generation_results
        
        # Advanced analysis using ADK
        logging.info("Phase 2: Advanced analysis with ADK agents...")
        master_recommendations = self.generate_master_recommendations(all_results)
        
        # Generate comprehensive report
        logging.info("Phase 3: Generating advanced research report...")
        report_path = self._generate_advanced_report(
            baseline_metrics, all_results, master_recommendations
        )
        
        logging.info(f"Advanced research session completed. Report: {report_path}")
        return report_path
    
    def _generate_advanced_report(self, baseline: Dict, results: List[ExperimentResult], 
                                 recommendations: Dict) -> str:
        """Generate advanced research report with ADK insights."""
        
        report_lines = [
            "# Advanced RAG System Research Report",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "## Executive Summary",
            "",
            recommendations.get("summary", "Analysis in progress..."),
            "",
            "## Master Recommendations",
            "",
            recommendations.get("master_recommendations", "Recommendations in progress..."),
            "",
            "## Detailed Analysis",
            "",
            "### Embedding Models Analysis",
            "",
            recommendations.get("embedding_analysis", {}).get("analysis", "Analysis pending..."),
            "",
            "### Generation Models Analysis", 
            "",
            recommendations.get("generation_analysis", {}).get("analysis", "Analysis pending..."),
            "",
            "## Baseline Performance",
            "",
            f"```json\n{json.dumps(baseline, ensure_ascii=False, indent=2)}\n```",
            "",
            "## All Experiment Results",
            ""
        ]
        
        # Add detailed experiment results
        for i, result in enumerate(results, 1):
            report_lines.extend([
                f"### Experiment {i}: {result.config.name}",
                f"**Description:** {result.config.description}",
                f"**Timestamp:** {result.timestamp}",
                f"**Execution Time:** {result.execution_time:.2f}s",
                "",
                "**Configuration:**",
                f"```json\n{json.dumps(result.config.__dict__, ensure_ascii=False, indent=2)}\n```",
                "",
                "**Metrics:**",
                f"```json\n{json.dumps(result.metrics, ensure_ascii=False, indent=2)}\n```",
                "",
                f"**Notes:** {result.notes}",
                "",
                "---",
                ""
            ])
        
        # Save report
        report_content = "\n".join(report_lines)
        report_path = f"research_results/advanced_research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        os.makedirs("research_results", exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        
        return report_path
    
    def _extract_best_model(self, results: List[ExperimentResult], metric: str) -> str:
        """Extract the best performing model based on a metric."""
        if not results:
            return "No results available"
        
        best_result = max(results, key=lambda r: r.metrics.get(metric, 0))
        return (best_result.config.embedding_model or 
                best_result.config.generation_model or 
                "Unknown model")
    
    def _fallback_embedding_analysis(self, results: List[ExperimentResult]) -> Dict[str, Any]:
        """Fallback analysis when ADK is not available."""
        best_model = self._extract_best_model(results, "faithfulness")
        return {
            "analysis": f"Best embedding model: {best_model} (ADK analysis not available)",
            "best_model": best_model,
            "raw_results": [result.__dict__ for result in results]
        }
    
    def _fallback_generation_analysis(self, results: List[ExperimentResult]) -> Dict[str, Any]:
        """Fallback analysis when ADK is not available.""" 
        best_model = self._extract_best_model(results, "faithfulness")
        return {
            "analysis": f"Best generation model: {best_model} (ADK analysis not available)",
            "best_model": best_model, 
            "raw_results": [result.__dict__ for result in results]
        }
    
    def _fallback_master_recommendations(self, results: List[ExperimentResult]) -> Dict[str, Any]:
        """Fallback recommendations when ADK is not available."""
        return {
            "master_recommendations": "ADK master coordination not available. Please configure Google API key.",
            "summary": f"Completed {len(results)} experiments. Manual analysis required.",
            "embedding_analysis": {},
            "generation_analysis": {}
        }
    
    def _create_executive_summary(self, results: List[ExperimentResult]) -> str:
        """Create executive summary of research findings."""
        total_experiments = len(results)
        successful_experiments = len([r for r in results if "error" not in r.metrics])
        
        return f"""
        Research session completed with {total_experiments} experiments.
        {successful_experiments} experiments completed successfully.
        
        Key focus areas: Embedding model optimization, Generation model selection, 
        Performance analysis for Hebrew medical RAG system.
        
        Detailed recommendations available in the full report sections.
        """


def main():
    """Main entry point for ADK research orchestrator."""
    orchestrator = ADKResearchOrchestrator()
    
    try:
        if ADK_AVAILABLE:
            report_path = orchestrator.run_advanced_research_session()
            print(f"Advanced research completed. Report available at: {report_path}")
        else:
            print("Google ADK not available. Please configure API key in .env file.")
            print("Falling back to basic research agent...")
            report_path = orchestrator.core_agent.run_full_research_session()
            print(f"Basic research completed. Report available at: {report_path}")
            
    except KeyboardInterrupt:
        print("\nResearch session interrupted by user.")
    except Exception as e:
        logging.error(f"Research orchestration failed: {e}")
        raise


if __name__ == "__main__":
    main()

