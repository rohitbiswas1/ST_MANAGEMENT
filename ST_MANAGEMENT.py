import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple
import re
import logging
import requests
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- AI Assistant Configuration ---
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

# Load the API key from Streamlit's secrets with error handling
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except (FileNotFoundError, KeyError):
    API_KEY = None # Set to None if key is not found

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
        (90, 'A+', 4.0),
        (80, 'A', 3.7),
        (70, 'B+', 3.3),
        (60, 'B', 3.0),
        (50, 'C', 2.0),
        (40, 'D', 1.0),
        (0, 'F', 0.0)
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
        if not os.path.exists(data_dir):
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
                os.rename(self.data_file, backup_name)
                st.warning(f"Corrupted data file backed up as {backup_name}")
    
    def _validate_and_fix_data(self) -> None:
        """Validate and fix existing data structure"""
        updated = False
        valid_students = {}
        
        for student_id, details in self.students.items():
            try:
                if not all(key in details for key in ['name', 'course', 'marks']):
                    logger.warning(f"Skipping invalid student record: {student_id}")
                    continue
                
                if 'gpa' not in details:
                    _, gpa = self.calculate_grade(details.get('marks', 0))
                    details['gpa'] = gpa
                    updated = True
                
                if 'grade' not in details:
                    grade, _ = self.calculate_grade(details.get('marks', 0))
                    details['grade'] = grade
                    updated = True
                
                if 'last_updated' not in details:
                    details['last_updated'] = details.get('date_added', datetime.now().isoformat())
                    updated = True
                
                details.setdefault('email', '')
                details.setdefault('phone', '')
                
                valid_students[student_id] = details
                
            except Exception as e:
                logger.error(f"Error validating student {student_id}: {e}")
                continue
        
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
                backup_file = f"{self.data_file}.backup"
                os.replace(self.data_file, backup_file)
            
            os.replace(temp_file, self.data_file)
            return True
            
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            st.error(f"Failed to save data: {e}")
            return False
    
    def calculate_grade(self, marks: float) -> Tuple[str, float]:
        """Calculate grade and GPA based on marks using defined scale"""
        try:
            marks = float(marks)
            for min_marks, grade, gpa in self.GRADE_SCALE:
                if marks >= min_marks:
                    return grade, gpa
            return 'F', 0.0
        except (ValueError, TypeError):
            return 'F', 0.0
    
    def validate_email(self, email: str) -> bool:
        """Validate email format with improved regex"""
        if not email or email.strip() == '':
            return True  # Email is optional
        
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email.strip()))
    
    def validate_phone(self, phone: str) -> bool:
        """Validate phone number format"""
        if not phone or phone.strip() == '':
            return True  # Phone is optional
        
        cleaned = re.sub(r'[\s\-\(\)\.]+', '', phone.strip())
        pattern = r'^(\+\d{1,3})?\d{10,15}$'
        return bool(re.match(pattern, cleaned))
    
    def generate_student_id(self) -> str:
        """Generate unique student ID with improved logic"""
        if not self.students:
            return "STU001"
        
        existing_nums = []
        for sid in self.students.keys():
            if sid.startswith("STU") and len(sid) >= 6:
                try:
                    num = int(sid[3:])
                    existing_nums.append(num)
                except ValueError:
                    continue
        
        next_num = max(existing_nums, default=0) + 1
        return f"STU{next_num:03d}"
    
    def add_student(self, name: str, course: str, marks: float, email: str = "", phone: str = "") -> Tuple[bool, str]:
        """Add a new student record with comprehensive validation"""
        try:
            if not name or not name.strip():
                return False, "Student name is required!"
            
            if not course or not course.strip():
                return False, "Course is required!"
            
            try:
                marks = float(marks)
                if not (0 <= marks <= 100):
                    return False, "Marks must be between 0 and 100!"
            except (ValueError, TypeError):
                return False, "Invalid marks format!"
            
            if email and not self.validate_email(email):
                return False, "Invalid email format!"
            
            if phone and not self.validate_phone(phone):
                return False, "Invalid phone number format!"
            
            name_clean = name.strip().lower()
            for existing_student in self.students.values():
                if existing_student['name'].lower() == name_clean:
                    return False, f"Student '{name.strip()}' already exists!"
            
            student_id = self.generate_student_id()
            grade, gpa = self.calculate_grade(marks)
            current_time = datetime.now().isoformat()
            
            self.students[student_id] = {
                'name': name.strip().title(),
                'course': course.strip().title(),
                'marks': marks,
                'grade': grade,
                'gpa': gpa,
                'email': email.strip().lower() if email else "",
                'phone': phone.strip() if phone else "",
                'date_added': current_time,
                'last_updated': current_time
            }
            
            if self.save_data():
                return True, f"Student '{name.strip()}' added successfully! ID: {student_id}, Grade: {grade}, GPA: {gpa}"
            else:
                return False, "Failed to save student data!"
            
        except Exception as e:
            logger.error(f"Error adding student: {e}")
            return False, f"Error adding student: {str(e)}"
    
    def update_student(self, student_id: str, **kwargs) -> Tuple[bool, str]:
        """Update existing student record with validation"""
        try:
            if student_id not in self.students:
                return False, f"Student ID {student_id} not found!"
            
            student = self.students[student_id].copy()
            
            for field, value in kwargs.items():
                if value is None:
                    continue
                
                if field == 'marks':
                    try:
                        marks = float(value)
                        if not (0 <= marks <= 100):
                            return False, "Marks must be between 0 and 100!"
                        student['marks'] = marks
                        grade, gpa = self.calculate_grade(marks)
                        student['grade'] = grade
                        student['gpa'] = gpa
                    except (ValueError, TypeError):
                        return False, "Invalid marks format!"
                        
                elif field == 'email':
                    if value and not self.validate_email(str(value)):
                        return False, "Invalid email format!"
                    student['email'] = str(value).strip().lower() if value else ""
                    
                elif field == 'phone':
                    if value and not self.validate_phone(str(value)):
                        return False, "Invalid phone number format!"
                    student['phone'] = str(value).strip() if value else ""
                    
                elif field in ['name', 'course']:
                    if not str(value).strip():
                        return False, f"{field.title()} cannot be empty!"
                    
                    if field == 'name':
                        name_clean = str(value).strip().lower()
                        for sid, existing_student in self.students.items():
                            if sid != student_id and existing_student['name'].lower() == name_clean:
                                return False, f"Student '{value}' already exists!"
                    
                    student[field] = str(value).strip().title()
            
            student['last_updated'] = datetime.now().isoformat()
            self.students[student_id] = student
            
            if self.save_data():
                return True, f"Student '{student['name']}' updated successfully!"
            else:
                return False, "Failed to save updated data!"
            
        except Exception as e:
            logger.error(f"Error updating student: {e}")
            return False, f"Error updating student: {str(e)}"
    
    def delete_student(self, student_id: str) -> Tuple[bool, str]:
        """Delete a student record"""
        try:
            if student_id not in self.students:
                return False, f"Student ID {student_id} not found!"
            
            student_name = self.students[student_id]['name']
            del self.students[student_id]
            
            if self.save_data():
                return True, f"Student '{student_name}' deleted successfully!"
            else:
                return False, "Failed to save after deletion!"
            
        except Exception as e:
            logger.error(f"Error deleting student: {e}")
            return False, f"Error deleting student: {str(e)}"
    
    def search_students(self, search_term: str, search_field: str = "all") -> pd.DataFrame:
        """Search students with improved error handling"""
        df = self.get_students_dataframe()
        if df.empty or not search_term or not search_term.strip():
            return df
        
        search_term = str(search_term).strip().lower()
        
        try:
            if search_field == "name":
                mask = df['Name'].str.lower().str.contains(search_term, na=False, regex=False)
            elif search_field == "course":
                mask = df['Course'].str.lower().str.contains(search_term, na=False, regex=False)
            elif search_field == "grade":
                mask = df['Grade'].str.lower().str.contains(search_term, na=False, regex=False)
            else:  # search all fields
                mask = (
                    df['Name'].str.lower().str.contains(search_term, na=False, regex=False) |
                    df['Course'].str.lower().str.contains(search_term, na=False, regex=False) |
                    df['Grade'].str.lower().str.contains(search_term, na=False, regex=False) |
                    df['Student ID'].str.lower().str.contains(search_term, na=False, regex=False)
                )
            
            return df[mask]
            
        except Exception as e:
            logger.error(f"Error in search: {e}")
            return pd.DataFrame()
    
    def get_students_dataframe(self) -> pd.DataFrame:
        """Convert student data to pandas DataFrame with error handling"""
        if not self.students:
            return pd.DataFrame()
        
        try:
            data = []
            for student_id, details in self.students.items():
                row = {
                    'Student ID': student_id,
                    'Name': details.get('name', ''),
                    'Course': details.get('course', ''),
                    'Marks': details.get('marks', 0),
                    'Grade': details.get('grade', ''),
                    'GPA': details.get('gpa', 0.0),
                    'Email': details.get('email', ''),
                    'Phone': details.get('phone', ''),
                    'Date Added': details.get('date_added', ''),
                    'Last Updated': details.get('last_updated', details.get('date_added', ''))
                }
                data.append(row)
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Error creating dataframe: {e}")
            return pd.DataFrame()
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics with error handling"""
        if not self.students:
            return {
                'total_students': 0, 'highest_marks': 0, 'lowest_marks': 0,
                'average_marks': 0, 'median_marks': 0, 'average_gpa': 0,
                'highest_gpa': 0, 'lowest_gpa': 0, 'grade_distribution': {},
                'course_distribution': {}, 'performance_levels': {},
                'passed': 0, 'failed': 0, 'pass_rate': 0
            }
        
        try:
            marks_list = [details.get('marks', 0) for details in self.students.values()]
            gpa_list = [details.get('gpa', 0.0) for details in self.students.values()]
            
            grade_count = {}
            for details in self.students.values():
                grade = details.get('grade', 'F')
                grade_count[grade] = grade_count.get(grade, 0) + 1
            
            course_count = {}
            for details in self.students.values():
                course = details.get('course', 'Unknown')
                course_count[course] = course_count.get(course, 0) + 1
            
            performance_levels = {
                'Excellent (90-100)': sum(1 for m in marks_list if m >= 90),
                'Good (70-89)': sum(1 for m in marks_list if 70 <= m < 90),
                'Average (50-69)': sum(1 for m in marks_list if 50 <= m < 70),
                'Poor (<50)': sum(1 for m in marks_list if m < 50)
            }
            
            passed = sum(1 for m in marks_list if m >= 40)
            failed = len(marks_list) - passed
            
            return {
                'total_students': len(self.students),
                'highest_marks': max(marks_list) if marks_list else 0,
                'lowest_marks': min(marks_list) if marks_list else 0,
                'average_marks': sum(marks_list) / len(marks_list) if marks_list else 0,
                'median_marks': sorted(marks_list)[len(marks_list)//2] if marks_list else 0,
                'average_gpa': sum(gpa_list) / len(gpa_list) if gpa_list else 0,
                'highest_gpa': max(gpa_list) if gpa_list else 0,
                'lowest_gpa': min(gpa_list) if gpa_list else 0,
                'grade_distribution': grade_count,
                'course_distribution': course_count,
                'performance_levels': performance_levels,
                'passed': passed,
                'failed': failed,
                'pass_rate': (passed / len(marks_list)) * 100 if marks_list else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {e}")
            return {}

@st.cache_data
def load_css():
    """Load custom CSS styles with mobile optimizations"""
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 600;
    }
    .ai-assistant-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .ai-chat-box {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
        color: #333;
        max-height: 200px;
        overflow-y: auto;
    }
    .success-message {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
        color: #155724;
    }
    .chat-message {
        padding: 8px 12px;
        margin-bottom: 8px;
        border-radius: 18px;
        max-width: 100%;
        line-height: 1.5;
    }
    .user-message {
        background-color: #007bff;
        color: white;
        text-align: right;
        margin-left: 20%;
    }
    .assistant-message {
        background-color: #f1f1f1;
        color: #333;
        text-align: left;
        margin-right: 20%;
    }
    
    /* --- Mobile Optimization --- */
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.8rem;
            margin-bottom: 1rem;
        }

        /* Make metric font size smaller on mobile */
        [data-testid="stMetricValue"] {
            font-size: 1.5rem;
        }

        [data-testid="stMetricLabel"] {
            font-size: 0.9rem;
        }

        /* Reduce padding for a tighter layout */
        .ai-assistant-container, .success-message {
            padding: 1rem;
        }

        /* Improve readability in expanders */
        .st-expander p {
            font-size: 0.9rem;
        }
    }
    </style>
    """

def initialize_session_state():
    """Initialize session state variables"""
    if 'sgm' not in st.session_state:
        st.session_state.sgm = StudentGradeManager()
    
    if 'ai_messages' not in st.session_state:
        st.session_state.ai_messages = []
    
    if 'show_ai_assistant' not in st.session_state:
        st.session_state.show_ai_assistant = False
    
    confirmations = ['confirm_clear_all', 'confirm_reset']
    for conf in confirmations:
        if conf not in st.session_state:
            st.session_state[conf] = False

def safe_execute(func, *args, **kwargs):
    """Safely execute a function with error handling"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.error(f"Error in {func.__name__}: {e}")
        return None

def generate_ai_response(prompt: str, data: str) -> str:
    """Generates a response from the AI assistant with improved error handling"""
    # Return a helpful message if the API key is not configured
    if not API_KEY:
        return "AI Assistant is disabled. Please configure the `GEMINI_API_KEY` in your Streamlit secrets to enable this feature."

    try:
        if not data or data.strip() == "Empty DataFrame":
            student_info = "No student data available."
        else:
            student_info = data
        
        full_prompt = f"""You are an educational AI assistant helping teachers analyze student performance data. 

STUDENT DATA:
{student_info}

QUESTION: {prompt}

Please provide a clear, helpful answer based on the student data. If no data is available, let the user know they need to add students first."""

        payload = {
            "contents": [{
                "parts": [{
                    "text": full_prompt
                }]
            }]
        }

        headers = {'Content-Type': 'application/json'}

        for attempt in range(3):
            try:
                response = requests.post(
                    f"{API_URL}?key={API_KEY}",
                    json=payload,
                    headers=headers,
                    timeout=30
                )
                response.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                logger.warning(f"API attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    raise
            
        result = response.json()
        
        if 'candidates' in result and len(result['candidates']) > 0:
            candidate = result['candidates'][0]
            if 'content' in candidate and 'parts' in candidate['content']:
                parts = candidate['content']['parts']
                if parts and 'text' in parts[0]:
                    return parts[0]['text'].strip()
        
        logger.error(f"Unexpected API response format: {result}")
        return "I apologize, but I'm having trouble processing your request right now. Please try again."

    except Exception as e:
        logger.error(f"AI response generation error: {e}")
        return f"I'm sorry, I encountered an error while processing your request: {str(e)}"

def render_ai_assistant_slot(sgm):
    """Render the AI Assistant slot at the top of the page"""
    # Disable the AI assistant if the API key is not available
    if not API_KEY:
        return

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🤖 AI Assistant - Ask Questions About Your Data", 
                     type="primary", 
                     use_container_width=True,
                     key="ai_toggle"):
            st.session_state.show_ai_assistant = not st.session_state.show_ai_assistant
    
    if st.session_state.show_ai_assistant:
        st.markdown("""
        <div class="ai-assistant-container">
            <h3>🤖 AI Educational Assistant</h3>
            <p>Ask me anything about student performance, trends, predictions, or get insights from your data!</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.ai_messages:
            st.markdown('<div class="ai-chat-box">', unsafe_allow_html=True)
            for msg in st.session_state.ai_messages[-3:]:
                if msg["role"] == "user":
                    st.markdown(f'<div class="chat-message user-message">👤 {msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-message assistant-message">🤖 {msg["content"][:300]}{"..." if len(msg["content"]) > 300 else ""}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with st.form("ai_quick_chat", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])
            with col1:
                user_prompt = st.text_input("Ask your question:", 
                                          placeholder="e.g., 'What's the average performance?'",
                                          key="ai_quick_input")
            with col2:
                submitted = st.form_submit_button("Send", type="primary")
            
            if submitted and user_prompt:
                st.session_state.ai_messages.append({"role": "user", "content": user_prompt})
                
                with st.spinner("🤖 Analyzing..."):
                    df_data_str = sgm.get_students_dataframe().to_string()
                    ai_response = generate_ai_response(user_prompt, df_data_str)
                    st.session_state.ai_messages.append({"role": "assistant", "content": ai_response})
                
                st.rerun()
        
        col1, col2, col3, col4 = st.columns(4)
        quick_questions = [
            ("📊 Performance", "Give me a performance summary"),
            ("⚠️ At Risk", "Which students are at risk?"),
            ("🏆 Top Performers", "Who are the top performers?"),
            ("💡 Recommendations", "Give me recommendations to improve performance.")
        ]
        
        buttons = [col1, col2, col3, col4]
        for i, (button_text, question) in enumerate(quick_questions):
            with buttons[i]:
                if st.button(button_text, key=f"quick_{i}", use_container_width=True):
                    st.session_state.ai_messages.append({"role": "user", "content": question})
                    with st.spinner("🤖 Analyzing..."):
                        df_data_str = sgm.get_students_dataframe().to_string()
                        ai_response = generate_ai_response(question, df_data_str)
                        st.session_state.ai_messages.append({"role": "assistant", "content": ai_response})
                    st.rerun()
        
        st.markdown("---")

def dashboard_page(sgm):
    """Enhanced Dashboard page"""
    st.header("📊 Dashboard Overview")
    
    if not sgm.students:
        st.info("Welcome! Start by adding students to see your dashboard come to life.")
        
        st.subheader("Quick Add Student")
        with st.form("quick_add", clear_on_submit=True):
            col1, col2 = st.columns([1,1])
            
            with col1:
                name = st.text_input("Name", placeholder="Enter student name")
                course = st.text_input("Course", placeholder="Enter course name")
            
            with col2:
                marks = st.number_input("Marks", min_value=0.0, max_value=100.0, value=0.0)
                if marks > 0:
                    grade, gpa = sgm.calculate_grade(marks)
                    st.info(f"Grade: {grade} | GPA: {gpa}")
            
            if st.form_submit_button("Add Student", type="primary"):
                if name and course:
                    success, message = sgm.add_student(name, course, marks)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.error("Name and Course are required!")
        return
    
    stats = sgm.get_statistics()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total Students", stats['total_students'])
    with col2: st.metric("Average Marks", f"{stats['average_marks']:.1f}")
    with col3: st.metric("Pass Rate", f"{stats['pass_rate']:.1f}%")
    with col4: st.metric("Average GPA", f"{stats['average_gpa']:.2f}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if stats['grade_distribution']:
            fig = px.pie(
                values=list(stats['grade_distribution'].values()),
                names=list(stats['grade_distribution'].keys()),
                title="Grade Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if stats['performance_levels']:
            fig = px.bar(
                x=list(stats['performance_levels'].keys()),
                y=list(stats['performance_levels'].values()),
                title="Performance Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Recent Students")
    df = sgm.get_students_dataframe()
    if not df.empty:
        cols_to_show = ['Student ID', 'Name', 'Course', 'Marks', 'Grade']
        recent_df = df.sort_values(by='Date Added', ascending=False).head(5)[cols_to_show]
        st.dataframe(recent_df, use_container_width=True)

def add_student_page(sgm):
    """Page for adding new student records."""
    st.header("➕ Add New Student")
    
    if sgm.students:
        stats = sgm.get_statistics()
        col1, col2, col3 = st.columns(3)
        with col1: st.info(f"👥 **Total:** {stats['total_students']}")
        with col2: st.info(f"📊 **Avg Marks:** {stats['average_marks']:.1f}")
        with col3: st.info(f"✅ **Pass Rate:** {stats['pass_rate']:.1f}%")
    
    st.markdown("---")
    
    with st.form("add_student_form", clear_on_submit=True):
        st.subheader("📝 Student Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Required Fields**")
            name = st.text_input("Student Name *", placeholder="Enter full name")
            course = st.text_input("Course/Subject *", placeholder="e.g., Computer Science")
            marks = st.number_input("Marks *", min_value=0.0, max_value=100.0, step=0.1, value=0.0)
        
        with col2:
            st.markdown("**Optional Fields**")
            email = st.text_input("Email", placeholder="student@example.com")
            phone = st.text_input("Phone", placeholder="+91 9876543210")
            
            if marks > 0:
                grade, gpa = sgm.calculate_grade(marks)
                st.markdown(f"""
                <div class="success-message">
                    <strong>📊 Calculated Results:</strong><br>
                    🎯 <strong>Grade:</strong> {grade}<br>
                    📈 <strong>GPA:</strong> {gpa}<br>
                    {'✅ Pass' if marks >= 40 else '❌ Fail'}
                </div>
                """, unsafe_allow_html=True)
        
        submitted = st.form_submit_button("➕ Add Student", type="primary", use_container_width=True)
        
        if submitted:
            with st.spinner("Adding student..."):
                success, message = sgm.add_student(name, course, marks, email, phone)
            
            if success:
                st.success(message)
                st.balloons()
                st.info(f"🆔 Next Student ID will be: **{sgm.generate_student_id()}**")
            else:
                st.error(f"❌ **Error:** {message}")

def manage_students_page(sgm):
    """Page for managing existing students."""
    st.header("✏️ Manage Students")
    
    if not sgm.students:
        st.info("📝 No students found. Add some students first!")
        return
    
    df = sgm.get_students_dataframe()
    
    col1, col2, col3 = st.columns(3)
    with col1: sort_by = st.selectbox("Sort by:", ["Name", "Marks", "Grade", "Course", "Date Added"])
    with col2: sort_order = st.selectbox("Sort order:", ["Ascending", "Descending"])
    with col3: filter_grade = st.selectbox("Filter by Grade:", ["All"] + sorted(df['Grade'].unique().tolist()))
    
    filtered_df = df.copy()
    if filter_grade != "All":
        filtered_df = filtered_df[filtered_df['Grade'] == filter_grade]
    
    ascending = sort_order == "Ascending"
    filtered_df = filtered_df.sort_values(sort_by, ascending=ascending)
    
    st.subheader(f"👥 Student List ({len(filtered_df)} students)")
    
    if filtered_df.empty:
        st.warning("No students match the current filters.")
        return
    
    for _, row in filtered_df.iterrows():
        expander_key = f"expander_{row['Student ID']}"
        with st.expander(f"🎓 {row['Name']} - {row['Student ID']} (Grade: {row['Grade']})"):
            
            st.markdown(f"""
            <p>
            <b>ID:</b> <code>{row['Student ID']}</code><br>
            <b>Name:</b> {row['Name']}<br>
            <b>Course:</b> {row['Course']}<br>
            <b>Marks:</b> {row['Marks']} | <b>Grade:</b> {row['Grade']} (GPA: {row['GPA']})<br>
            <b>Email:</b> {row['Email'] if row['Email'] else 'Not provided'}<br>
            <b>Phone:</b> {row['Phone'] if row['Phone'] else 'Not provided'}<br>
            <b>Added:</b> {row['Date Added']}<br>
            <b>Updated:</b> {row['Last Updated']}
            </p>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            with st.form(f"edit_form_{row['Student ID']}"):
                st.subheader("Edit Student Information")
                edit_col1, edit_col2 = st.columns(2)
                
                with edit_col1:
                    new_name = st.text_input("Name", value=row['Name'])
                    new_course = st.text_input("Course", value=row['Course'])
                    new_marks = st.number_input("Marks", min_value=0.0, max_value=100.0, value=float(row['Marks']))
                
                with edit_col2:
                    new_email = st.text_input("Email", value=row['Email'])
                    new_phone = st.text_input("Phone", value=row['Phone'])
                    
                    if new_marks != row['Marks']:
                        new_grade, new_gpa = sgm.calculate_grade(new_marks)
                        st.info(f"New Grade: **{new_grade}** (GPA: {new_gpa})")
                
                update_col, delete_col = st.columns(2)
                
                with update_col:
                    if st.form_submit_button("💾 Save Changes", type="primary", use_container_width=True):
                        success, message = sgm.update_student(
                            row['Student ID'], name=new_name, course=new_course, 
                            marks=new_marks, email=new_email, phone=new_phone
                        )
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
                
                with delete_col:
                    if st.form_submit_button("🗑️ Delete Student", type="secondary", use_container_width=True):
                        success, message = sgm.delete_student(row['Student ID'])
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)

def search_students_page(sgm):
    """Page for searching students."""
    st.header("🔍 Search & Filter Students")
    
    if not sgm.students:
        st.info("🔍 No students found. Add some students first!")
        return
    
    col1, col2 = st.columns([3, 1])
    with col1:
        search_term = st.text_input("🔍 Search Students", 
                                  placeholder="Enter name, course, grade, or ID...")
    with col2:
        search_field = st.selectbox("Search in:", ["All Fields", "Name", "Course", "Grade"])
    
    with st.expander("🔧 Advanced Filters"):
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        df = sgm.get_students_dataframe()
        
        with filter_col1:
            grade_filter = st.multiselect("Filter by Grade:", options=sorted(df['Grade'].unique()))
        with filter_col2:
            course_filter = st.multiselect("Filter by Course:", options=sorted(df['Course'].unique()))
        with filter_col3:
            marks_range = st.slider("Marks Range:", 0, 100, (0, 100))
    
    search_field_map = {"All Fields": "all", "Name": "name", "Course": "course", "Grade": "grade"}
    results_df = sgm.search_students(search_term, search_field_map[search_field])
    
    # Apply advanced filters
    if grade_filter: results_df = results_df[results_df['Grade'].isin(grade_filter)]
    if course_filter: results_df = results_df[results_df['Course'].isin(course_filter)]
    results_df = results_df[(results_df['Marks'] >= marks_range[0]) & (results_df['Marks'] <= marks_range[1])]
    
    st.subheader(f"📊 Search Results ({len(results_df)} students found)")
    if results_df.empty:
        st.warning("🔍 No students match your search criteria. Try adjusting your filters.")
    else:
        st.dataframe(results_df, use_container_width=True)
        
        if st.button("📤 Export Search Results", type="secondary", use_container_width=True):
            csv_data = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📄 Download as CSV", data=csv_data,
                file_name=f"search_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

def analytics_page(sgm):
    """Page for analytics and reports."""
    st.header("📊 Analytics & Reports")
    
    if not sgm.students:
        st.info("📈 No data available for analytics. Add some students first!")
        return
    
    stats = sgm.get_statistics()
    df = sgm.get_students_dataframe()
    
    st.subheader("🎯 Executive Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total Students", stats['total_students'])
    with col2: st.metric("Average Marks", f"{stats['average_marks']:.1f}%")
    with col3: st.metric("Pass Rate", f"{stats['pass_rate']:.1f}%")
    with col4: st.metric("Average GPA", f"{stats['average_gpa']:.2f}")
    
    tab1, tab2, tab3 = st.tabs(["📈 Performance", "📚 Courses", "🎯 Grades"])
    
    with tab1:
        st.subheader("📈 Performance Analysis")
        col1, col2 = st.columns(2)
        with col1:
            perf_data = stats['performance_levels']
            fig = px.pie(values=list(perf_data.values()), names=list(perf_data.keys()), 
                        title="Performance Level Distribution")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.histogram(df, x='Marks', nbins=20, title="Marks Distribution")
            fig.add_vline(x=stats['average_marks'], line_dash="dash", line_color="red", 
                         annotation_text=f"Avg: {stats['average_marks']:.1f}")
            fig.add_vline(x=stats['median_marks'], line_dash="dash", line_color="green", 
                         annotation_text=f"Med: {stats['median_marks']:.1f}")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("📚 Course Analysis")
        if not df.empty:
            course_stats = df.groupby('Course').agg(
                Students=('Course', 'count'),
                Avg_Marks=('Marks', 'mean'),
                Avg_GPA=('GPA', 'mean')
            ).reset_index().sort_values(by="Students", ascending=False)
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(course_stats, x='Course', y='Students', title="Enrollment by Course")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.bar(course_stats, x='Course', y='Avg_Marks', title="Avg Performance by Course")
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("📊 Course Performance Summary")
            st.dataframe(course_stats, use_container_width=True)
    
    with tab3:
        st.subheader("🎯 Grade Analysis")
        col1, col2 = st.columns(2)
        with col1:
            grade_data = stats['grade_distribution']
            fig = px.pie(values=list(grade_data.values()), names=list(grade_data.keys()), 
                        title="Grade Distribution")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            pass_fail_data = {'Passed': stats['passed'], 'Failed': stats['failed']}
            fig = px.pie(values=list(pass_fail_data.values()), names=list(pass_fail_data.keys()), 
                        title="Pass/Fail Distribution", color_discrete_sequence=['#28a745', '#dc3545'])
            st.plotly_chart(fig, use_container_width=True)

def import_export_page(sgm):
    """Page for data import/export operations."""
    st.header("📤 Import & Export Data")
    
    tab1, tab2 = st.tabs(["📤 Export Data", "📥 Import Data"])
    
    with tab1:
        st.subheader("📊 Export Student Data")
        
        if not sgm.students:
            st.info("📝 No data to export. Add some students first!")
        else:
            df = sgm.get_students_dataframe()
            export_df = df.copy()

            st.subheader("📋 Export Preview")
            st.dataframe(export_df.head(5), use_container_width=True)
            
            filename = f"students_{datetime.now().strftime('%Y%m%d_%H%M')}"
            
            col1, col2 = st.columns(2)
            with col1:
                csv_data = export_df.to_csv(index=False).encode('utf-8')
                st.download_button("📄 Download as CSV", csv_data, file_name=f"{filename}.csv", 
                                  mime="text/csv", type="primary", use_container_width=True)
            with col2:
                json_data = export_df.to_json(orient='records', indent=2).encode('utf-8')
                st.download_button("📄 Download as JSON", json_data, file_name=f"{filename}.json", 
                                  mime="application/json", type="primary", use_container_width=True)
    
    with tab2:
        st.subheader("📥 Import Student Data")
        st.info("**📋 Import Requirements:**\n- File must be CSV with `Name`, `Course`, `Marks`.\n- Optional columns: `Email`, `Phone`.")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file:
            try:
                import_df = pd.read_csv(uploaded_file)
                st.subheader("📊 Import Preview")
                st.dataframe(import_df.head(), use_container_width=True)
                
                required_cols = {'Name', 'Course', 'Marks'}
                if not required_cols.issubset(import_df.columns):
                    st.error(f"❌ Missing required columns: {', '.join(required_cols - set(import_df.columns))}")
                else:
                    valid_data_df = import_df.dropna(subset=['Name', 'Course', 'Marks']).copy()
                    valid_data_df = valid_data_df[(valid_data_df['Marks'] >= 0) & (valid_data_df['Marks'] <= 100)]
                    
                    if valid_data_df.empty:
                        st.warning("No valid rows found to import.")
                    elif st.button(f"📥 Import {len(valid_data_df)} Students", type="primary"):
                        with st.spinner("Importing data..."):
                            success_count = 0
                            error_messages = []
                            for _, row in valid_data_df.iterrows():
                                success, message = sgm.add_student(
                                    name=str(row['Name']), course=str(row['Course']),
                                    marks=float(row['Marks']),
                                    email=str(row.get('Email', '')),
                                    phone=str(row.get('Phone', ''))
                                )
                                if success:
                                    success_count += 1
                                else:
                                    error_messages.append(f"Row '{row['Name']}': {message}")
                                
                        st.success(f"✅ Successfully imported {success_count} students!")
                        if error_messages:
                            with st.expander(f"⚠️ {len(error_messages)} students could not be imported. See errors."):
                                for error in error_messages:
                                    st.write(f"- {error}")
                        st.balloons()
                        st.rerun()
            except Exception as e:
                st.error(f"❌ Error processing file: {e}")

def settings_page(sgm):
    """Page for system settings."""
    st.header("⚙️ Settings & Configuration")
    
    tab1, tab2, tab3 = st.tabs(["🗃️ Data Management", "⚙️ System", "ℹ️ About"])
    
    with tab1:
        st.subheader("🗃️ Data Management")
        
        if sgm.students:
            stats = sgm.get_statistics()
            st.info(f"📈 **Database Stats:** {stats['total_students']} students in `{sgm.data_file}`")
        else:
            st.info("📝 No student data found.")
            
        st.markdown("---")
        
        st.subheader("⚠️ Dangerous Operations")
        st.warning("These operations cannot be undone. Please be careful!")
        
        if st.button("🗑️ Clear All Data", type="secondary", use_container_width=True):
            st.session_state['confirm_clear_all'] = True
            
        if st.session_state.get('confirm_clear_all'):
            st.error("⚠️ **Warning:** This will delete ALL student data permanently!")
            if st.button("I'm Sure, Clear ALL Data", type="primary"):
                sgm.students = {}
                sgm.save_data()
                st.session_state['confirm_clear_all'] = False
                st.success("✅ All data cleared successfully!")
                st.rerun()
    
    with tab2:
        st.subheader("⚙️ System Configuration")
        st.markdown("### Grading Scale")
        st.dataframe(pd.DataFrame(sgm.GRADE_SCALE, columns=['Min Marks', 'Grade', 'GPA']), use_container_width=True)
        
        st.markdown("### AI Assistant Status")
        if API_KEY:
            st.success("✅ AI Assistant is configured and ready.")
        else:
            st.error("❌ AI Assistant is disabled. Add `GEMINI_API_KEY` to your Streamlit secrets.")
        
    with tab3:
        st.subheader("ℹ️ About This System")
        st.markdown("""
        ### Student Grade Management System
        
        **Version:** 2.2 (FIXED)
        **Built with:** Python, Streamlit, Plotly, Google Gemini AI
        - Complete student record management
        - Advanced analytics and visualizations
        - AI-powered insights and recommendations
        - Data import/export functionality
        - Responsive design for Desktop & Mobile
        
        **Created for:** Educational institutions and instructors.
        """)

def main():
    """Main application function"""
    st.markdown(load_css(), unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">🎓 Student Grade Management System</h1>', unsafe_allow_html=True)
    
    # Show a persistent warning at the top if the API key is not set.
    if not API_KEY:
        st.warning("⚠️ **AI Assistant is disabled.** To enable it, please add your `GEMINI_API_KEY` to your Streamlit secrets. See the 'Settings' page for more info.")

    initialize_session_state()
    sgm = st.session_state.sgm
    
    render_ai_assistant_slot(sgm)
    
    # Sidebar
    st.sidebar.title("📋 Navigation")
    
    total_students = len(sgm.students)
    st.sidebar.metric("Total Students", total_students)
    
    if sgm.students:
        stats = sgm.get_statistics()
        st.sidebar.metric("Average Marks", f"{stats['average_marks']:.1f}")
        st.sidebar.metric("Pass Rate", f"{stats['pass_rate']:.1f}%")
    
    pages = {
        "Dashboard": "🏠", "Add Student": "➕", "Manage Students": "✏️",
        "Search Students": "🔍", "Analytics": "📊", "Import/Export": "📤",
        "Settings": "⚙️"
    }
    
    # Using selectbox for navigation to avoid rendering issues.
    selected_page = st.sidebar.selectbox(
        "Choose a page:",
        list(pages.keys()),
        format_func=lambda x: f"{pages[x]} {x}"
    )
    
    # Page routing dictionary for cleaner code
    page_functions = {
        "Dashboard": dashboard_page,
        "Add Student": add_student_page,
        "Manage Students": manage_students_page,
        "Search Students": search_students_page,
        "Analytics": analytics_page,
        "Import/Export": import_export_page,
        "Settings": settings_page,
    }
    
    # Execute the selected page's function
    if selected_page in page_functions:
        safe_execute(page_functions[selected_page], sgm)
    else:
        st.error("Page not found!")

if __name__ == "__main__":
    main()
