<<<<<<< HEAD
#!/usr/bin/env python
"""
BiztelAI Data Pipeline Runner

This script runs the data pipeline for the BiztelAI dataset, loading the
configuration from a JSON file and executing the pipeline with the specified
processors.
"""
import os
import sys
import argparse
import logging
import pandas as pd
from typing import Dict, Any

from data_processor import ProcessorFactory

def main():
    """Main function to run the data pipeline."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the BiztelAI data pipeline')
    parser.add_argument('--config', '-c', type=str, default='pipeline_config.json',
                        help='Path to pipeline configuration file')
    parser.add_argument('--input', '-i', type=str,
                        help='Input file path (overrides config)')
    parser.add_argument('--output', '-o', type=str,
                        help='Output file path (overrides config)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting BiztelAI data pipeline using config: {args.config}")
    
    try:
        # Create data pipeline from configuration
        pipeline = ProcessorFactory.from_config_file(args.config)
        
        # Override input/output if provided as command-line arguments
        if args.input:
            pipeline.config['input_file'] = args.input
            logger.info(f"Overriding input file: {args.input}")
        
        if args.output:
            pipeline.config['output_file'] = args.output
            logger.info(f"Overriding output file: {args.output}")
        
        # Get input file from pipeline config
        input_file = pipeline.config.get('input_file')
        if not input_file:
            logger.error("No input file specified in config or command line")
            return 1
        
        # Check if input file exists
        if not os.path.exists(input_file):
            logger.error(f"Input file not found: {input_file}")
            return 1
        
        # Load initial data
        logger.info(f"Loading data from: {input_file}")
        initial_data = pd.DataFrame()  # This will be populated by the first processor
        
        # Process data through pipeline
        logger.info("Running data pipeline...")
        result_data = pipeline.process(initial_data)
        
        # Save results
        output_file = pipeline.config.get('output_file')
        if output_file:
            logger.info(f"Saving processed data to: {output_file}")
            
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            # Save based on file extension
            file_ext = os.path.splitext(output_file)[1].lower()
            if file_ext == '.csv':
                result_data.to_csv(output_file, index=False)
            elif file_ext == '.json':
                result_data.to_json(output_file, orient='records', lines=True)
            elif file_ext == '.parquet':
                result_data.to_parquet(output_file, index=False)
            elif file_ext == '.pkl' or file_ext == '.pickle':
                result_data.to_pickle(output_file)
            else:
                logger.warning(f"Unknown file extension: {file_ext}, defaulting to CSV")
                result_data.to_csv(output_file, index=False)
        
        # Print pipeline metadata
        logger.info("Pipeline execution complete.")
        
        # Print summary statistics
        logger.info(f"Input shape: {initial_data.shape if hasattr(initial_data, 'shape') else 'N/A'}")
        logger.info(f"Output shape: {result_data.shape}")
        
        # Print processor execution times if available
        for proc_meta in pipeline.metadata.get('processors', []):
            if 'timing' in proc_meta and 'execution_time' in proc_meta['timing']:
                logger.info(f"Processor {proc_meta['name']} execution time: "
                           f"{proc_meta['timing']['execution_time']:.4f} seconds")
        
        return 0
        
    except Exception as e:
        logger.exception(f"Error running pipeline: {e}")
        return 1

if __name__ == "__main__":
=======
#!/usr/bin/env python
"""
BiztelAI Data Pipeline Runner

This script runs the data pipeline for the BiztelAI dataset, loading the
configuration from a JSON file and executing the pipeline with the specified
processors.
"""
import os
import sys
import argparse
import logging
import pandas as pd
from typing import Dict, Any

from data_processor import ProcessorFactory

def main():
    """Main function to run the data pipeline."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the BiztelAI data pipeline')
    parser.add_argument('--config', '-c', type=str, default='pipeline_config.json',
                        help='Path to pipeline configuration file')
    parser.add_argument('--input', '-i', type=str,
                        help='Input file path (overrides config)')
    parser.add_argument('--output', '-o', type=str,
                        help='Output file path (overrides config)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting BiztelAI data pipeline using config: {args.config}")
    
    try:
        # Create data pipeline from configuration
        pipeline = ProcessorFactory.from_config_file(args.config)
        
        # Override input/output if provided as command-line arguments
        if args.input:
            pipeline.config['input_file'] = args.input
            logger.info(f"Overriding input file: {args.input}")
        
        if args.output:
            pipeline.config['output_file'] = args.output
            logger.info(f"Overriding output file: {args.output}")
        
        # Get input file from pipeline config
        input_file = pipeline.config.get('input_file')
        if not input_file:
            logger.error("No input file specified in config or command line")
            return 1
        
        # Check if input file exists
        if not os.path.exists(input_file):
            logger.error(f"Input file not found: {input_file}")
            return 1
        
        # Load initial data
        logger.info(f"Loading data from: {input_file}")
        initial_data = pd.DataFrame()  # This will be populated by the first processor
        
        # Process data through pipeline
        logger.info("Running data pipeline...")
        result_data = pipeline.process(initial_data)
        
        # Save results
        output_file = pipeline.config.get('output_file')
        if output_file:
            logger.info(f"Saving processed data to: {output_file}")
            
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            # Save based on file extension
            file_ext = os.path.splitext(output_file)[1].lower()
            if file_ext == '.csv':
                result_data.to_csv(output_file, index=False)
            elif file_ext == '.json':
                result_data.to_json(output_file, orient='records', lines=True)
            elif file_ext == '.parquet':
                result_data.to_parquet(output_file, index=False)
            elif file_ext == '.pkl' or file_ext == '.pickle':
                result_data.to_pickle(output_file)
            else:
                logger.warning(f"Unknown file extension: {file_ext}, defaulting to CSV")
                result_data.to_csv(output_file, index=False)
        
        # Print pipeline metadata
        logger.info("Pipeline execution complete.")
        
        # Print summary statistics
        logger.info(f"Input shape: {initial_data.shape if hasattr(initial_data, 'shape') else 'N/A'}")
        logger.info(f"Output shape: {result_data.shape}")
        
        # Print processor execution times if available
        for proc_meta in pipeline.metadata.get('processors', []):
            if 'timing' in proc_meta and 'execution_time' in proc_meta['timing']:
                logger.info(f"Processor {proc_meta['name']} execution time: "
                           f"{proc_meta['timing']['execution_time']:.4f} seconds")
        
        return 0
        
    except Exception as e:
        logger.exception(f"Error running pipeline: {e}")
        return 1

if __name__ == "__main__":
>>>>>>> 5d96285edcdaab24a8fcf7efb1d7ec4f7b04bb22
    sys.exit(main()) 