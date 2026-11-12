from pydantic import BaseModel, Field

class Student(BaseModel):
    name: str

new_student = Student(name=32)
 
print(new_student)