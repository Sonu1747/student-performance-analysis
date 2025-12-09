-- Student Performance Monitoring & Dropout Prediction Database Schema
-- This schema supports tracking student performance, attendance, and engagement

-- Students table
CREATE TABLE students (
    student_id INT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    age INT NOT NULL,
    gender VARCHAR(10) NOT NULL,
    class VARCHAR(20) NOT NULL,
    enrollment_date DATE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Grades table
CREATE TABLE grades (
    grade_id INT PRIMARY KEY AUTO_INCREMENT,
    student_id INT NOT NULL,
    subject VARCHAR(50) NOT NULL,
    score DECIMAL(5,2) NOT NULL,
    term VARCHAR(20) NOT NULL,
    exam_date DATE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES students(student_id),
    INDEX idx_student_subject (student_id, subject),
    INDEX idx_term (term)
);

-- Attendance table
CREATE TABLE attendance (
    attendance_id INT PRIMARY KEY AUTO_INCREMENT,
    student_id INT NOT NULL,
    date DATE NOT NULL,
    status ENUM('Present', 'Absent', 'Late') NOT NULL,
    period VARCHAR(20),
    subject VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES students(student_id),
    INDEX idx_student_date (student_id, date),
    INDEX idx_date (date)
);

-- Engagement table
CREATE TABLE engagement (
    engagement_id INT PRIMARY KEY AUTO_INCREMENT,
    student_id INT NOT NULL,
    login_time TIMESTAMP NOT NULL,
    activity_type ENUM('quiz', 'video', 'assignment', 'forum', 'resource') NOT NULL,
    duration_minutes INT,
    subject VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES students(student_id),
    INDEX idx_student_login (student_id, login_time),
    INDEX idx_activity_type (activity_type)
);

-- Dropout predictions table
CREATE TABLE dropout_predictions (
    prediction_id INT PRIMARY KEY AUTO_INCREMENT,
    student_id INT NOT NULL,
    prediction_date DATE NOT NULL,
    dropout_risk_score DECIMAL(3,2) NOT NULL,
    risk_level ENUM('Low', 'Medium', 'High', 'Critical') NOT NULL,
    model_version VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES students(student_id),
    INDEX idx_student_prediction (student_id, prediction_date),
    INDEX idx_risk_level (risk_level)
);

-- Sample data insertion
INSERT INTO students (student_id, name, age, gender, class, enrollment_date) VALUES
(1, 'John Smith', 16, 'Male', 'Grade 10A', '2024-01-15'),
(2, 'Sarah Johnson', 17, 'Female', 'Grade 11B', '2024-01-15'),
(3, 'Mike Davis', 16, 'Male', 'Grade 10A', '2024-01-15'),
(4, 'Emily Brown', 17, 'Female', 'Grade 11B', '2024-01-15'),
(5, 'David Wilson', 18, 'Male', 'Grade 12A', '2024-01-15');

-- Sample grades data
INSERT INTO grades (student_id, subject, score, term, exam_date) VALUES
(1, 'Mathematics', 85.5, 'Term 1', '2024-03-15'),
(1, 'English', 78.0, 'Term 1', '2024-03-16'),
(1, 'Science', 92.0, 'Term 1', '2024-03-17'),
(2, 'Mathematics', 76.5, 'Term 1', '2024-03-15'),
(2, 'English', 88.0, 'Term 1', '2024-03-16'),
(2, 'Science', 81.5, 'Term 1', '2024-03-17'),
(3, 'Mathematics', 65.0, 'Term 1', '2024-03-15'),
(3, 'English', 72.0, 'Term 1', '2024-03-16'),
(3, 'Science', 68.5, 'Term 1', '2024-03-17');

-- Sample attendance data
INSERT INTO attendance (student_id, date, status, period, subject) VALUES
(1, '2024-03-01', 'Present', 'Morning', 'Mathematics'),
(1, '2024-03-02', 'Present', 'Morning', 'English'),
(1, '2024-03-03', 'Present', 'Morning', 'Science'),
(2, '2024-03-01', 'Present', 'Morning', 'Mathematics'),
(2, '2024-03-02', 'Absent', 'Morning', 'English'),
(2, '2024-03-03', 'Present', 'Morning', 'Science'),
(3, '2024-03-01', 'Present', 'Morning', 'Mathematics'),
(3, '2024-03-02', 'Present', 'Morning', 'English'),
(3, '2024-03-03', 'Absent', 'Morning', 'Science');

-- Sample engagement data
INSERT INTO engagement (student_id, login_time, activity_type, duration_minutes, subject) VALUES
(1, '2024-03-01 09:00:00', 'quiz', 30, 'Mathematics'),
(1, '2024-03-01 14:00:00', 'video', 45, 'Science'),
(1, '2024-03-02 10:00:00', 'assignment', 60, 'English'),
(2, '2024-03-01 09:15:00', 'quiz', 25, 'Mathematics'),
(2, '2024-03-01 15:00:00', 'video', 30, 'Science'),
(3, '2024-03-01 09:30:00', 'quiz', 20, 'Mathematics'),
(3, '2024-03-02 11:00:00', 'assignment', 40, 'English');