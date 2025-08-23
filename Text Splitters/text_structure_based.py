from langchain.text_splitter import RecursiveCharacterTextSplitter

text = """
Gandhinagar Institute of Skill Development (GISD) Approved by: Gandhinagar University 
Master of Technology 
Duration: 2 year 
• M.Tech - Mechanical Engineering (Thermal) - Intake: 9 
• M.Tech - Mechanical Engineering (CAD/CAM) - Intake: 9 
• M.Tech - Computer Engineering (Software Engineering) - Intake: 18 
• M.Tech - Computer Science and Engineering (Cyber Security) - Intake: 18 
• M.Tech - Computer Science and Engineering (Artificial Intelligence and Machine Learning) - Intake: 18 
• M.Tech - Civil Engineering (Environmental Engineering) - Intake: 18 
• M.Tech - Civil (Structural Engineering) - Intake: 18 
• M.Tech - Civil (Construction Engineering and Management) - Intake: 18 
• M.Tech - Electronics and Communication Technology (VLSI System Design) - Intake: 18 • M.Tech - Energy Engineering - Intake: 18 
Bachelor of Technology 
Duration: 4 year 
• B.Tech - Civil Engineering - Intake: 120 
• B.Tech - Computer Engineering - Intake: 180 
• B.Tech - Electrical Engineering - Intake: 60 
• B.Tech - Electronics and Communication Engineering - Intake: 60 
• B.Tech - Information Technology - Intake: 180 
• B.Tech - Mechanical Engineering - Intake: 120 
• B.Tech - Computer Science and Engineering - Intake: 420 
• B.Tech - Computer Science and Engineering (Cyber Security) - Intake: 60
• B.Tech - Computer Science and Engineering (Artificial Intelligence) - Intake: 120 • B.Tech - Robotics and Automation - Intake: 60 
• B.Tech - Energy Engineering - Intake: 60 
Diploma in Engineering 
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 400,
)

result = splitter.split_text(text)

print(result[1])