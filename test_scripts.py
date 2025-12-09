"""
Test Script for Student Performance Monitoring Project
This script tests the basic functionality of all components.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    try:
        # Test basic imports
        import pandas as pd
        import numpy as np
        print("✓ pandas and numpy imported successfully")
        
        # Test sklearn imports (if available)
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import classification_report
            print("✓ scikit-learn imported successfully")
        except ImportError:
            print("⚠ scikit-learn not available (install with: pip install scikit-learn)")
        
        # Test other imports
        try:
            import yaml
            print("✓ PyYAML imported successfully")
        except ImportError:
            print("⚠ PyYAML not available (install with: pip install pyyaml)")
        
        try:
            import schedule
            print("✓ schedule imported successfully")
        except ImportError:
            print("⚠ schedule not available (install with: pip install schedule)")
        
        return True
        
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        return False

def test_data_creation():
    """Test data creation and basic operations."""
    print("\nTesting data creation...")
    
    try:
        # Create sample data
        np.random.seed(42)
        n_samples = 100
        
        data = {
            'student_id': range(1, n_samples + 1),
            'name': [f'Student_{i}' for i in range(1, n_samples + 1)],
            'age': np.random.randint(15, 20, n_samples),
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'class': np.random.choice(['Grade 10A', 'Grade 11B', 'Grade 12A'], n_samples),
            'avg_score': np.random.normal(75, 15, n_samples),
            'attendance_rate': np.random.beta(2, 1, n_samples),
            'total_activities': np.random.poisson(20, n_samples)
        }
        
        df = pd.DataFrame(data)
        print(f"✓ Created sample dataset with {len(df)} records")
        
        # Test basic operations
        df['risk_score'] = np.where(
            (df['avg_score'] < 60) | (df['attendance_rate'] < 0.7),
            1, 0
        )
        
        high_risk_count = df['risk_score'].sum()
        print(f"✓ Identified {high_risk_count} high-risk students")
        
        return True
        
    except Exception as e:
        print(f"✗ Data creation test failed: {e}")
        return False

def test_file_structure():
    """Test if all required files exist."""
    print("\nTesting file structure...")
    
    required_files = [
        'sql/schema.sql',
        'python/data_pipeline.py',
        'python/feature_engineering.py',
        'python/model_training.py',
        'python/predict.py',
        'python/automation.py',
        'config/config.yaml',
        'requirements.txt',
        'powerbi/dashboard_guide.md'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"✓ {file_path} exists")
    
    if missing_files:
        print(f"✗ Missing files: {missing_files}")
        return False
    else:
        print("✓ All required files exist")
        return True

def test_config_loading():
    """Test configuration loading."""
    print("\nTesting configuration loading...")
    
    try:
        import yaml
        
        if os.path.exists('config/config.yaml'):
            with open('config/config.yaml', 'r') as file:
                config = yaml.safe_load(file)
            
            required_sections = ['database', 'email', 'automation', 'powerbi']
            missing_sections = [section for section in required_sections if section not in config]
            
            if missing_sections:
                print(f"✗ Missing config sections: {missing_sections}")
                return False
            else:
                print("✓ Configuration loaded successfully")
                return True
        else:
            print("✗ Configuration file not found")
            return False
            
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        return False

def test_sql_schema():
    """Test SQL schema file."""
    print("\nTesting SQL schema...")
    
    try:
        with open('sql/schema.sql', 'r') as file:
            schema_content = file.read()
        
        required_tables = ['students', 'grades', 'attendance', 'engagement', 'dropout_predictions']
        missing_tables = [table for table in required_tables if table not in schema_content.lower()]
        
        if missing_tables:
            print(f"✗ Missing tables in schema: {missing_tables}")
            return False
        else:
            print("✓ SQL schema contains all required tables")
            return True
            
    except Exception as e:
        print(f"✗ SQL schema test failed: {e}")
        return False

def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("STUDENT PERFORMANCE MONITORING - TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Imports", test_imports),
        ("Data Creation", test_data_creation),
        ("Configuration", test_config_loading),
        ("SQL Schema", test_sql_schema)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:20} : {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Project is ready to use.")
    else:
        print("⚠ Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
