"""
Configuration module for Model Distillation
"""

from .distillation_config import DistillationConfig
from .teacher_config import TeacherConfig
from .student_config import StudentConfig

__all__ = ["DistillationConfig", "TeacherConfig", "StudentConfig"]
