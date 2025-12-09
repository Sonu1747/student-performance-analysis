"""
Run data pipeline with error handling
"""

import sys
import os
sys.path.append('python')

try:
    from data_pipeline import StudentDataPipeline
    
    print("=" * 60)
    print("RUNNING DATA PIPELINE")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = StudentDataPipeline()
    print("✓ Pipeline initialized")
    
    # Run pipeline
    print("Starting data processing...")
    result = pipeline.run_pipeline()
    
    print(f"✓ Pipeline completed successfully!")
    print(f"✓ Processed {len(result)} student records")
    print("\nSample of processed data:")
    print(result.head())
    
    print(f"\nData saved to: data/processed/student_data_processed.csv")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

