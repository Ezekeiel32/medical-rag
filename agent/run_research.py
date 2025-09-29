#!/usr/bin/env python3
"""
Main entry point for RAG Research Session

This script orchestrates a comprehensive research session to optimize 
the Hebrew medical RAG system using both traditional methods and 
Google ADK-powered agents.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add the parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from research_agent import RAGResearchAgent
from adk_research_orchestrator import ADKResearchOrchestrator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('research_session.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def check_environment() -> bool:
    """Check if the environment is properly configured."""
    logger.info("Checking environment configuration...")
    
    # Check if we're in the right directory
    if not os.path.exists("../src/cli.py"):
        logger.error("Must be run from the agent/ directory within the RAG project")
        return False
    
    # Check if OCR data exists
    ocr_dir = "/home/chezy/rag_medical/ocr_out"
    if not os.path.exists(ocr_dir):
        logger.error(f"OCR data directory not found: {ocr_dir}")
        return False
    
    required_files = [
        "structured_documents.jsonl",
        "documents_index.json"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(ocr_dir, file)):
            missing_files.append(file)
    
    if missing_files:
        logger.warning(f"Missing OCR files: {missing_files}")
        logger.info("These will be generated during the research session if needed.")
    
    # Check if .env file exists
    if os.path.exists(".env"):
        logger.info("Found .env file")
        with open(".env", "r") as f:
            content = f.read()
            if "YOUR_ACTUAL_API_KEY_HERE" in content:
                logger.warning("Google API key not configured in .env file")
                logger.info("ADK features will be disabled, falling back to basic research")
            else:
                logger.info("Google API key appears to be configured")
    else:
        logger.warning("No .env file found. ADK features will be disabled.")
    
    logger.info("Environment check completed")
    return True


def run_research_session(mode: str = "auto", ocr_dir: str = None) -> str:
    """Run the research session in the specified mode."""
    
    if not check_environment():
        logger.error("Environment check failed")
        return None
    
    ocr_dir = ocr_dir or "/home/chezy/rag_medical/ocr_out"
    
    logger.info(f"Starting research session in {mode} mode...")
    logger.info(f"Using OCR directory: {ocr_dir}")
    
    try:
        if mode == "adk" or (mode == "auto" and os.getenv("GOOGLE_API_KEY")):
            logger.info("Using Google ADK orchestration")
            orchestrator = ADKResearchOrchestrator(ocr_dir)
            report_path = orchestrator.run_advanced_research_session()
        else:
            logger.info("Using basic research agent")
            agent = RAGResearchAgent(ocr_dir)
            report_path = agent.run_full_research_session()
        
        return report_path
        
    except KeyboardInterrupt:
        logger.info("Research session interrupted by user")
        return None
    except Exception as e:
        logger.error(f"Research session failed: {e}")
        raise


def main():
    """Main entry point with command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run RAG system research session",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_research.py                    # Auto-detect mode
  python run_research.py --mode basic       # Use basic research agent only
  python run_research.py --mode adk         # Force ADK mode
  python run_research.py --ocr-dir /path    # Custom OCR directory
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["auto", "basic", "adk"],
        default="auto",
        help="Research mode: auto (detect), basic (no ADK), adk (force ADK)"
    )
    
    parser.add_argument(
        "--ocr-dir",
        default="/home/chezy/rag_medical/ocr_out",
        help="Path to OCR output directory"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("="*60)
    logger.info("RAG System Research Session Starting")
    logger.info("="*60)
    
    try:
        report_path = run_research_session(args.mode, args.ocr_dir)
        
        if report_path:
            logger.info("="*60)
            logger.info("Research Session Completed Successfully!")
            logger.info(f"Report available at: {report_path}")
            logger.info("="*60)
            
            print(f"\n🎉 Research completed! Report saved to: {report_path}")
            print("\nNext steps:")
            print("1. Review the research report for key findings")
            print("2. Implement recommended optimizations")
            print("3. Test the optimized system")
            print("4. Monitor performance improvements")
        else:
            logger.error("Research session did not complete successfully")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Fatal error in research session: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

