"""
Script to create SQLite database with sample data
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def create_database():
    """Create SQLite database with sample data."""
    
    # Connect to SQLite database (creates if doesn't exist)
    conn = sqlite3.connect('student_performance.db')
    cursor = conn.cursor()
    
    # Read and execute schema
    try:
        with open('sql/schema.sql', 'r') as file:
            schema_sql = file.read()
        
        # Split by semicolon and execute each statement
        statements = schema_sql.split(';')
        for statement in statements:
            statement = statement.strip()
            if statement and not statement.startswith('--'):
                try:
                    cursor.execute(statement)
                except sqlite3.OperationalError as e:
                    if "already exists" not in str(e):
                        print(f"Warning: {e}")
        
        conn.commit()
        print("Database schema created successfully!")
    except Exception as e:
        print(f"Error reading schema: {e}")
        # Create basic tables manually
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            student_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            age INTEGER NOT NULL,
            gender TEXT NOT NULL,
            class TEXT NOT NULL,
            enrollment_date TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS grades (
            grade_id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            subject TEXT NOT NULL,
            score REAL NOT NULL,
            term TEXT NOT NULL,
            exam_date TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students(student_id)
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            attendance_id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            status TEXT NOT NULL,
            period TEXT,
            subject TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students(student_id)
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS engagement (
            engagement_id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            login_time TEXT NOT NULL,
            activity_type TEXT NOT NULL,
            duration_minutes INTEGER,
            subject TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students(student_id)
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS dropout_predictions (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            prediction_date TEXT NOT NULL,
            dropout_risk_score REAL NOT NULL,
            risk_level TEXT NOT NULL,
            model_version TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students(student_id)
        )
        """)
        
        conn.commit()
        print("Basic database schema created!")
    
    # Insert initial sample data from schema
    initial_students = [
        (1, 'John Smith', 16, 'Male', 'Grade 10A', '2024-01-15'),
        (2, 'Sarah Johnson', 17, 'Female', 'Grade 11B', '2024-01-15'),
        (3, 'Mike Davis', 16, 'Male', 'Grade 10A', '2024-01-15'),
        (4, 'Emily Brown', 17, 'Female', 'Grade 11B', '2024-01-15'),
        (5, 'David Wilson', 18, 'Male', 'Grade 12A', '2024-01-15')
    ]
    
    cursor.executemany(
        "INSERT OR IGNORE INTO students (student_id, name, age, gender, class, enrollment_date) VALUES (?, ?, ?, ?, ?, ?)",
        initial_students
    )
    
    initial_grades = [
        (1, 'Mathematics', 85.5, 'Term 1', '2024-03-15'),
        (1, 'English', 78.0, 'Term 1', '2024-03-16'),
        (1, 'Science', 92.0, 'Term 1', '2024-03-17'),
        (2, 'Mathematics', 76.5, 'Term 1', '2024-03-15'),
        (2, 'English', 88.0, 'Term 1', '2024-03-16'),
        (2, 'Science', 81.5, 'Term 1', '2024-03-17'),
        (3, 'Mathematics', 65.0, 'Term 1', '2024-03-15'),
        (3, 'English', 72.0, 'Term 1', '2024-03-16'),
        (3, 'Science', 68.5, 'Term 1', '2024-03-17')
    ]
    
    cursor.executemany(
        "INSERT OR IGNORE INTO grades (student_id, subject, score, term, exam_date) VALUES (?, ?, ?, ?, ?)",
        initial_grades
    )
    
    initial_attendance = [
        (1, '2024-03-01', 'Present', 'Morning', 'Mathematics'),
        (1, '2024-03-02', 'Present', 'Morning', 'English'),
        (1, '2024-03-03', 'Present', 'Morning', 'Science'),
        (2, '2024-03-01', 'Present', 'Morning', 'Mathematics'),
        (2, '2024-03-02', 'Absent', 'Morning', 'English'),
        (2, '2024-03-03', 'Present', 'Morning', 'Science'),
        (3, '2024-03-01', 'Present', 'Morning', 'Mathematics'),
        (3, '2024-03-02', 'Present', 'Morning', 'English'),
        (3, '2024-03-03', 'Absent', 'Morning', 'Science')
    ]
    
    cursor.executemany(
        "INSERT OR IGNORE INTO attendance (student_id, date, status, period, subject) VALUES (?, ?, ?, ?, ?)",
        initial_attendance
    )
    
    initial_engagement = [
        (1, '2024-03-01 09:00:00', 'quiz', 30, 'Mathematics'),
        (1, '2024-03-01 14:00:00', 'video', 45, 'Science'),
        (1, '2024-03-02 10:00:00', 'assignment', 60, 'English'),
        (2, '2024-03-01 09:15:00', 'quiz', 25, 'Mathematics'),
        (2, '2024-03-01 15:00:00', 'video', 30, 'Science'),
        (3, '2024-03-01 09:30:00', 'quiz', 20, 'Mathematics'),
        (3, '2024-03-02 11:00:00', 'assignment', 40, 'English')
    ]
    
    cursor.executemany(
        "INSERT OR IGNORE INTO engagement (student_id, login_time, activity_type, duration_minutes, subject) VALUES (?, ?, ?, ?, ?)",
        initial_engagement
    )
    
    conn.commit()
    print("Initial sample data inserted!")
    
    # Generate additional sample data
    generate_sample_data(conn)
    
    conn.close()
    print("Database created with sample data!")

def generate_sample_data(conn):
    """Generate additional sample data for testing."""
    
    # Generate more students
    students_data = []
    for i in range(6, 51):  # Add 45 more students
        students_data.append((
            i,
            f'Student_{i}',
            random.randint(15, 19),
            random.choice(['Male', 'Female']),
            random.choice(['Grade 10A', 'Grade 10B', 'Grade 11A', 'Grade 11B', 'Grade 12A', 'Grade 12B']),
            '2024-01-15'
        ))
    
    cursor = conn.cursor()
    cursor.executemany(
        "INSERT OR IGNORE INTO students (student_id, name, age, gender, class, enrollment_date) VALUES (?, ?, ?, ?, ?, ?)",
        students_data
    )
    
    # Generate more grades
    subjects = ['Mathematics', 'English', 'Science', 'History', 'Geography', 'Computer Science']
    grades_data = []
    
    for student_id in range(1, 51):
        for subject in subjects:
            # Generate 2-4 grades per subject per student
            num_grades = random.randint(2, 4)
            for grade_num in range(num_grades):
                score = random.normalvariate(75, 15)  # Mean 75, std 15
                score = max(0, min(100, score))  # Clamp between 0-100
                
                grades_data.append((
                    student_id,
                    subject,
                    round(score, 1),
                    f'Term {random.randint(1, 3)}',
                    f'2024-{random.randint(3, 12):02d}-{random.randint(1, 28):02d}'
                ))
    
    cursor.executemany(
        "INSERT OR IGNORE INTO grades (student_id, subject, score, term, exam_date) VALUES (?, ?, ?, ?, ?)",
        grades_data
    )
    
    # Generate more attendance data
    attendance_data = []
    start_date = datetime(2024, 3, 1)
    
    for student_id in range(1, 51):
        for day in range(90):  # 90 days of attendance
            date = start_date + timedelta(days=day)
            status = random.choices(['Present', 'Absent', 'Late'], weights=[85, 10, 5])[0]
            
            attendance_data.append((
                student_id,
                date.strftime('%Y-%m-%d'),
                status,
                'Morning',
                random.choice(subjects)
            ))
    
    cursor.executemany(
        "INSERT OR IGNORE INTO attendance (student_id, date, status, period, subject) VALUES (?, ?, ?, ?, ?)",
        attendance_data
    )
    
    # Generate more engagement data
    engagement_data = []
    activity_types = ['quiz', 'video', 'assignment', 'forum', 'resource']
    
    for student_id in range(1, 51):
        # Generate 20-50 engagement activities per student
        num_activities = random.randint(20, 50)
        for activity_num in range(num_activities):
            login_time = datetime(2024, random.randint(3, 12), random.randint(1, 28), 
                                random.randint(8, 18), random.randint(0, 59))
            activity_type = random.choice(activity_types)
            duration = random.randint(10, 120)  # 10-120 minutes
            
            engagement_data.append((
                student_id,
                login_time.strftime('%Y-%m-%d %H:%M:%S'),
                activity_type,
                duration,
                random.choice(subjects)
            ))
    
    cursor.executemany(
        "INSERT OR IGNORE INTO engagement (student_id, login_time, activity_type, duration_minutes, subject) VALUES (?, ?, ?, ?, ?)",
        engagement_data
    )
    
    conn.commit()
    print("Sample data generated successfully!")
    
    # Print summary
    cursor.execute("SELECT COUNT(*) FROM students")
    print(f"Students: {cursor.fetchone()[0]}")
    
    cursor.execute("SELECT COUNT(*) FROM grades")
    print(f"Grades: {cursor.fetchone()[0]}")
    
    cursor.execute("SELECT COUNT(*) FROM attendance")
    print(f"Attendance records: {cursor.fetchone()[0]}")
    
    cursor.execute("SELECT COUNT(*) FROM engagement")
    print(f"Engagement records: {cursor.fetchone()[0]}")

if __name__ == "__main__":
    create_database()
