import os


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
print("BASE_DIR", BASE_DIR)


class Config:
    DEBUG = True
    # DEBUG = False
    # TESTING = False
    # ENV = 'production'


class ProjectConfig:
    templates_folder = os.path.join(BASE_DIR, "templates")
    static_folder = os.path.join(BASE_DIR, "static")
