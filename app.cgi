

from flask_app.py import app  
from wsgiref.handlers import CGIHandler

if __name__ == "__main__":
    CGIHandler().run(app)
