import os

BASE_DIR = r"C:\Users\Hp\OneDrive\Desktop\2nd Year\Sem-4\ML\Project"
RAW_DATA_PATH = os.path.join(BASE_DIR, "Data", "placement_predict_50k Dataset (2).csv")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "Data", "processed")
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "cleaned_data.csv")

PLOTS_DIR = os.path.join(BASE_DIR, "Output", "plot")

CATEGORICAL_COLS = [
    'Gender',
    'City',
    'CollegeTier',
    'Stream',
    'Specialisation',
    'Hostel',
    'HistoryOfBacklogs',
    'CGPA_Tier'
]

TARGETED_COLS = [
    'PlacementStatus',
    'Salary Package'
]

NUMERICAL_COLS = [
    'SGPA_Sem1',
    'SGPA_Sem2',
    'SGPA_Sem3',
    'SGPA_Sem4',
    'SGPA_Sem5',
    'SGPA_Sem6',
    'SGPA_Sem7',
    'SGPA_Sem8',
    'CGPA',
    'AttendancePercent',
    'Internships',
    'Projects',
    'Workshops',
    'Certifications',
    'Publications',
    'AptitudeTestScore',
    'SoftSkillsRating',
    'CodingTestScore',
    'MockInterviewScore',
    'ExtraCurricular'
]