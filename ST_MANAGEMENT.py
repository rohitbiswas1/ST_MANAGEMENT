import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
import plotly.express as px
from typing import Dict, Tuple
import re
import logging
import requests
import time
import traceback

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- AI Assistant API Configuration ---
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

# --- Securely load the API key from Streamlit's secrets ---
# This is the correct, secure way to handle your key.
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except (FileNotFoundError, KeyError):
    API_KEY = None # Allows the app to run without AI features if the key is not set.

# --- Page Configuration (Must be the first Streamlit command) ---
st.set_page_config(
    page_title="Student Grade Management System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# #############################################################################
# DATA MANAGEMENT CLASS
# #############################################################################
class StudentGradeManager:
    """Handles all student data operations: loading, saving, and manipulation."""
    
    GRADE_SCALE = [
        (90, 'A+', 4.0), (80, 'A', 3.7), (70, 'B+', 3.3), (60, 'B', 3.0),
        (50, 'C', 2.0), (40, 'D', 1.0), (0, 'F', 0.0)
    ]
    
    def __init__(self, data_file: str = "student_data.json"):
        self.data_file = data_file
        self.students: Dict[str, Dict] = {}
        self.load_data()
    
    def load_data(self) -> None:
        """Loads student data from the JSON file with error handling."""
        try:
            if os.path.exists(self.data_file) and os.path.getsize(self.data_file) > 0:
                with open(self.data_file, 'r', encoding='utf-8') as file:
                    self.students = json.load(file)
            else:
                self.students = {}
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading data file: {e}")
            self.students = {} # Start with a clean slate if file is corrupt.

    def save_data(self) -> bool:
        """Saves the current student data to the JSON file."""
        try:
            with open(self.data_file, 'w', encoding='utf-8') as file:
                json.dump(self.students, file, indent=4)
            return True
        except IOError as e:
            logger.error(f"Error saving data file: {e}")
            st.error(f"Critical Error: Failed to save data. {e}")
            return False

    def calculate_grade(self, marks: float) -> Tuple[str, float]:
        """Calculates grade and GPA from marks."""
        try:
            marks = float(marks)
            for min_marks, grade, gpa in self.GRADE_SCALE:
                if marks >= min_marks:
                    return grade, gpa
            return 'F', 0.0
        except (ValueError, TypeError):
            return 'F', 0.0

    def generate_student_id(self) -> str:
        """Generates a new, unique student ID."""
        if not self.students:
            return "STU001"
        # Extracts numbers from existing IDs (e.g., STU001 -> 1)
        nums = [int(sid[3:]) for sid in self.students.keys() if sid.startswith("STU") and sid[3:].isdigit()]
        return f"STU{max(nums, default=0) + 1:03d}"

    def add_student(self, name: str, course: str, marks: float, email: str = "", phone: str = "") -> Tuple[bool, str]:
        """Adds a new student to the records with validation."""
        if not name.strip() or not course.strip():
            return False, "Error: Name and Course are required fields."
        
        student_id = self.generate_student_id()
        grade, gpa = self.calculate_grade(marks)
        current_time = datetime.now().isoformat()
        
        self.students[student_id] = {
            'name': name.strip().title(), 'course': course.strip().title(), 'marks': marks,
            'grade': grade, 'gpa': gpa, 'email': email.strip().lower(), 'phone': phone.strip(),
            'date_added': current_time, 'last_updated': current_time
        }
        
        if self.save_data():
            return True, f"Success! Student '{name.strip()}' added with ID: {student_id}"
        return False, "Error: Failed to save the new student data."
    
    def get_students_dataframe(self) -> pd.DataFrame:
        """Converts the student dictionary to a pandas DataFrame."""
        if not self.students:
            return pd.DataFrame()
        df = pd.DataFrame.from_dict(self.students, orient='index')
        df.index.name = 'Student ID'
        return df.reset_index()

    def get_statistics(self) -> Dict:
        """Calculates key statistics from the student data."""
        if not self.students:
            return {'total_students': 0, 'average_marks': 0, 'pass_rate': 0}
        
        df = self.get_students_dataframe()
        passed_count = (df['marks'] >= 40).sum()
        total_count = len(df)
        
        return {
            'total_students': total_count,
            'average_marks': df['marks'].mean(),
            'pass_rate': (passed_count / total_count) * 100 if total_count > 0 else 0,
        }

# #############################################################################
# STYLES AND UTILITIES
# #############################################################################
def load_css():
    """Loads custom CSS for styling the application."""
    return """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 1.5rem;
            font-weight: 600;
        }
        /* Mobile-friendly styles */
        @media (max-width: 768px) {
            .main-header { font-size: 1.8rem; }
            [data-testid="stMetricValue"] { font-size: 1.5rem; }
        }
    </style>
    """

def initialize_session_state():
    """Initializes variables in the session state if they don't exist."""
    if 'sgm' not in st.session_state:
        st.session_state.sgm = StudentGradeManager()

# #############################################################################
# AI ASSISTANT FUNCTIONS
# #############################################################################
def generate_ai_response(prompt: str, data: pd.DataFrame) -> str:
    """Contacts the Gemini API to get an analysis of the student data."""
    if not API_KEY:
        return "AI Assistant is disabled. Please add your `GEMINI_API_KEY` to your Streamlit secrets."

    try:
        student_info = "No student data is available to analyze."
        if not data.empty:
            student_info = data.to_string()
        
        full_prompt = f"You are an expert educational analyst. Based on the following student data, please answer the user's question.\n\nDATA:\n{student_info}\n\nQUESTION: {prompt}"
        
        payload = {"contents": [{"parts": [{"text": full_prompt}]}]}
        headers = {'Content-Type': 'application/json'}
        response = requests.post(f"{API_URL}?key={API_KEY}", json=payload, headers=headers, timeout=20)
        response.raise_for_status() # Raises an error for bad responses (4xx or 5xx)
        
        result = response.json()
        return result['candidates'][0]['content']['parts'][0]['text'].strip()

    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        return f"Sorry, there was a network error communicating with the AI service: {e}"
    except (KeyError, IndexError):
        logger.error(f"Unexpected API response format: {response.text}")
        return "Sorry, the AI service returned an unexpected response. Please try again."

def render_ai_assistant_slot(sgm):
    """Renders the AI Assistant UI component."""
    if not API_KEY:
        return # Do not show the AI feature if the key is missing.

    with st.expander("🤖 AI Assistant - Get Insights on Your Data"):
        user_prompt = st.text_input("Ask a question:", placeholder="e.g., 'Which students need the most help?'")
        if st.button("Ask AI"):
            if user_prompt:
                with st.spinner("🧠 The AI is thinking..."):
                    df = sgm.get_students_dataframe()
                    ai_response = generate_ai_response(user_prompt, df)
                    st.info(ai_response)
            else:
                st.warning("Please enter a question to ask the AI.")

# #############################################################################
# PAGE RENDERING FUNCTIONS
# #############################################################################
def dashboard_page(sgm):
    """Renders the main dashboard page."""
    st.header("📊 Dashboard Overview")
    stats = sgm.get_statistics()
    
    if stats['total_students'] == 0:
        st.info("👋 Welcome! Your dashboard is ready. Add a student to see it in action.")
        return
        
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Students", f"{stats['total_students']:.0f}")
    col2.metric("Class Average", f"{stats['average_marks']:.1f}%")
    col3.metric("Pass Rate", f"{stats['pass_rate']:.1f}%")
    
    st.subheader("Recent Student Entries")
    st.dataframe(sgm.get_students_dataframe().tail(), use_container_width=True, hide_index=True)

def add_student_page(sgm):
    """Renders the page for adding a new student."""
    st.header("➕ Add a New Student Record")
    with st.form("add_student_form", clear_on_submit=True):
        st.write("Enter the student's details below.")
        name = st.text_input("Student Name *")
        course = st.text_input("Course/Subject *")
        marks = st.number_input("Marks (out of 100) *", min_value=0.0, max_value=100.0, step=0.5)
        email = st.text_input("Email Address (Optional)")
        
        submitted = st.form_submit_button("✓ Submit and Add Student")
        if submitted:
            if name and course:
                success, message = sgm.add_student(name, course, marks, email)
                if success:
                    st.success(message)
                else:
                    st.error(message)
            else:
                st.error("Please fill in all required fields (*).")

# #############################################################################
# MAIN APPLICATION LOGIC
# #############################################################################
def main():
    """The main function that runs the Streamlit application."""
    st.markdown(load_css(), unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">🎓 Student Grade Management System</h1>', unsafe_allow_html=True)
    
    # Display a clear, persistent warning if the API key is missing.
    if not API_KEY:
        st.warning("⚠️ **AI Assistant is offline.** To enable this feature, please add your `GEMINI_API_KEY` to your Streamlit secrets.")

    initialize_session_state()
    sgm = st.session_state.sgm
    
    # Render the AI assistant at the top of the page.
    render_ai_assistant_slot(sgm)
    
    # --- Sidebar Navigation ---
    st.sidebar.title("📋 Navigation")
    pages = {
        "Dashboard": "🏠",
        "Add Student": "➕",
    }
    
    # This selectbox is more stable than the radio button.
    selected_page = st.sidebar.selectbox(
        "Go to page:",
        list(pages.keys()),
        format_func=lambda page: f"{pages[page]} {page}" # Shows emoji and name
    )
    
    # --- Page Routing ---
    if selected_page == "Dashboard":
        dashboard_page(sgm)
    elif selected_page == "Add Student":
        add_student_page(sgm)

# --- Main Execution with Robust Error Handling ---
# This block will catch any startup errors and display them on the page.
if __name__ == "__main__":
    try:
        main()
    except Exception:
        st.error("💥 A critical error occurred. The application cannot continue.")
        st.code(traceback.format_exc())
