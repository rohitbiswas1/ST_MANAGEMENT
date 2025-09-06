import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Tuple
import re
import logging
import requests
import time
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- AI Assistant Configuration ---
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

# Securely load the API key from Streamlit's secrets
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except (FileNotFoundError, KeyError):
    API_KEY = None # App will run without AI features if key is not found

# Set page config
st.set_page_config(
    page_title="Student Grade Management System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StudentGradeManager:
    """Enhanced Student Grade Management System with improved error handling and validation"""
    
    GRADE_SCALE = [
        (90, 'A+', 4.0), (80, 'A', 3.7), (70, 'B+', 3.3), (60, 'B', 3.0),
        (50, 'C', 2.0), (40, 'D', 1.0), (0, 'F', 0.0)
    ]
    
    def __init__(self, data_file: str = "student_data.json"):
        """Initialize the Student Grade Management System"""
        self.data_file = data_file
        self.students: Dict[str, Dict] = {}
        self._ensure_data_directory()
        self.load_data()
    
    def _ensure_data_directory(self) -> None:
        """Ensure data directory exists"""
        data_dir = os.path.dirname(self.data_file) or "."
        os.makedirs(data_dir, exist_ok=True)
    
    def load_data(self) -> None:
        """Load student data from JSON file with enhanced error handling"""
        try:
            if os.path.exists(self.data_file) and os.path.getsize(self.data_file) > 0:
                with open(self.data_file, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    if isinstance(data, dict):
                        self.students = data
                        self._validate_and_fix_data()
                    else:
                        logger.warning("Invalid data format in file, initializing empty")
                        self.students = {}
            else:
                self.students = {}
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading data: {e}")
            self.students = {}
            if os.path.exists(self.data_file):
                backup_name = f"{self.data_file}.corrupted_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                try:
                    os.rename(self.data_file, backup_name)
                    st.warning(f"Corrupted data file backed up as {backup_name}")
                except OSError as backup_e:
                    logger.error(f"Could not back up corrupted file: {backup_e}")
    
    def _validate_and_fix_data(self) -> None:
        """Validate and fix existing data structure"""
        updated = False
        valid_students = {}
        for student_id, details in self.students.items():
            if not isinstance(details, dict) or not all(key in details for key in ['name', 'course', 'marks']):
                logger.warning(f"Skipping invalid student record: {student_id}")
                continue
            
            if 'gpa' not in details or 'grade' not in details:
                grade, gpa = self.calculate_grade(details.get('marks', 0))
                details['grade'] = grade
                details['gpa'] = gpa
                updated = True
            
            details.setdefault('last_updated', details.get('date_added', datetime.now().isoformat()))
            details.setdefault('email', '')
            details.setdefault('phone', '')
            valid_students[student_id] = details
        
        self.students = valid_students
        if updated:
            self.save_data()
    
    def save_data(self) -> bool:
        """Save student data to JSON file with error handling"""
        try:
            temp_file = f"{self.data_file}.tmp"
            with open(temp_file, 'w', encoding='utf-8') as file:
                json.dump(self.students, file, indent=2, ensure_ascii=False)
            
            if os.path.exists(self.data_file):
                os.replace(self.data_file, f"{self.data_file}.backup")
            os.replace(temp_file, self.data_file)
            return True
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            st.error(f"Failed to save data: {e}")
            return False
    
    def calculate_grade(self, marks: float) -> Tuple[str, float]:
        """Calculate grade and GPA based on marks"""
        try:
            marks = float(marks)
            for min_marks, grade, gpa in self.GRADE_SCALE:
                if marks >= min_marks:
                    return grade, gpa
            return 'F', 0.0
        except (ValueError, TypeError):
            return 'F', 0.0
    
    def validate_email(self, email: str) -> bool:
        if not email: return True
        return bool(re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email.strip()))
    
    def validate_phone(self, phone: str) -> bool:
        if not phone: return True
        cleaned = re.sub(r'[\s\-().]+', '', phone.strip())
        return bool(re.match(r'^(\+\d{1,3})?\d{10,15}$', cleaned))
    
    def generate_student_id(self) -> str:
        if not self.students: return "STU001"
        nums = [int(sid[3:]) for sid in self.students.keys() if sid.startswith("STU") and sid[3:].isdigit()]
        return f"STU{max(nums, default=0) + 1:03d}"

    def add_student(self, name: str, course: str, marks: float, email: str = "", phone: str = "") -> Tuple[bool, str]:
        if not name.strip() or not course.strip(): return False, "Name and Course are required."
        try:
            marks = float(marks)
            if not (0 <= marks <= 100): return False, "Marks must be between 0 and 100."
        except (ValueError, TypeError): return False, "Invalid marks format."
        if not self.validate_email(email): return False, "Invalid email format."
        if not self.validate_phone(phone): return False, "Invalid phone number format."

        if any(s['name'].lower() == name.strip().lower() for s in self.students.values()):
            return False, f"Student '{name.strip()}' already exists."

        student_id = self.generate_student_id()
        grade, gpa = self.calculate_grade(marks)
        current_time = datetime.now().isoformat()
        
        self.students[student_id] = {
            'name': name.strip().title(), 'course': course.strip().title(), 'marks': marks,
            'grade': grade, 'gpa': gpa, 'email': email.strip().lower(), 'phone': phone.strip(),
            'date_added': current_time, 'last_updated': current_time
        }
        
        if self.save_data():
            return True, f"Student '{name.strip()}' added! ID: {student_id}"
        return False, "Failed to save student data."

    def update_student(self, student_id: str, **kwargs) -> Tuple[bool, str]:
        if student_id not in self.students: return False, f"Student ID {student_id} not found."
        
        student = self.students[student_id].copy()
        
        # Validation logic remains the same... (Code omitted for brevity)
        
        student['last_updated'] = datetime.now().isoformat()
        self.students[student_id] = student
        
        if self.save_data(): return True, f"Student '{student['name']}' updated."
        return False, "Failed to save updated data."

    def delete_student(self, student_id: str) -> Tuple[bool, str]:
        if student_id not in self.students: return False, f"Student ID {student_id} not found."
        student_name = self.students.pop(student_id)['name']
        if self.save_data(): return True, f"Student '{student_name}' deleted."
        return False, "Failed to save after deletion."
    
    def get_students_dataframe(self) -> pd.DataFrame:
        if not self.students: return pd.DataFrame()
        return pd.DataFrame.from_dict(self.students, orient='index').reset_index().rename(columns={'index': 'Student ID'})
    
    def get_statistics(self) -> Dict:
        if not self.students: return {'total_students': 0, 'average_marks': 0, 'pass_rate': 0, 'average_gpa': 0}
        df = self.get_students_dataframe()
        marks = df['marks']
        passed = (marks >= 40).sum()
        return {
            'total_students': len(df), 'average_marks': marks.mean(),
            'pass_rate': (passed / len(df)) * 100, 'average_gpa': df['gpa'].mean(),
            'grade_distribution': df['grade'].value_counts().to_dict(),
            'performance_levels': {
                'Excellent (90-100)': (marks >= 90).sum(), 'Good (70-89)': ((marks >= 70) & (marks < 90)).sum(),
                'Average (50-69)': ((marks >= 50) & (marks < 70)).sum(), 'Poor (<50)': (marks < 50).sum()
            }, 'passed': passed, 'failed': len(df) - passed
        }

# I've removed @st.cache_data as a troubleshooting step.
def load_css():
    """Load custom CSS styles with mobile optimizations"""
    return """
    <style>
    /* CSS content remains the same... (Code omitted for brevity) */
    </style>
    """

def initialize_session_state():
    """Initialize session state variables"""
    if 'sgm' not in st.session_state:
        st.session_state.sgm = StudentGradeManager()
    if 'ai_messages' not in st.session_state:
        st.session_state.ai_messages = []
    # Other initializations...

# ... The rest of your page-rendering functions (dashboard_page, add_student_page, etc.) remain the same ...
# ... I have omitted them here for brevity but they should be in your file ...

def main():
    """Main application function"""
    st.markdown(load_css(), unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">🎓 Student Grade Management System</h1>', unsafe_allow_html=True)
    
    if not API_KEY:
        st.warning("⚠️ **AI Assistant is disabled.** To enable it, add `GEMINI_API_KEY` to your Streamlit secrets.")

    initialize_session_state()
    sgm = st.session_state.sgm
    
    render_ai_assistant_slot(sgm)
    
    st.sidebar.title("📋 Navigation")
    
    # ... Sidebar and page routing logic remains the same ...

# --- Main Execution with Error Handling ---
# This new block will catch any startup errors and display them on the screen.
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("💥 An unexpected error occurred while loading the application.")
        st.error(f"Error details: {e}")
        st.code(traceback.format_exc())
