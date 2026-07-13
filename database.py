"""
Database module — SQLite/MySQL query manager using SQLAlchemy.
Handles structured job data queries (filters, stats, etc.)
"""

import re
from typing import Optional
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import text as sql_text
import config

Base = declarative_base()


class Job(Base):
    """Job listing model for SQL database."""
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_title = Column(String(500), nullable=False)
    company_name = Column(String(500))
    location = Column(String(500))
    work_type = Column(String(100))
    salary_raw = Column(String(500))
    salary_min = Column(Float, nullable=True)
    salary_max = Column(Float, nullable=True)
    job_description = Column(Text)
    scrape_timestamp = Column(String(100))


def parse_salary(salary_str: str) -> tuple[Optional[float], Optional[float]]:
    """
    Parse salary string into min and max values.
    Examples:
        "Rp 10.000.000 – Rp 14.000.000 per month" -> (10000000, 14000000)
        "Rp 6.000.000 per month" -> (6000000, 6000000)
        "None" -> (None, None)
    """
    if not salary_str or salary_str == "None":
        return None, None

    # Find all numbers in the string (handle dot-separated thousands)
    numbers = re.findall(r'[\d]+(?:\.[\d]+)*', salary_str)
    parsed = []
    for num_str in numbers:
        # Remove dots used as thousand separators
        clean = num_str.replace(".", "")
        try:
            parsed.append(float(clean))
        except ValueError:
            continue

    # Filter out very small numbers (likely not salary)
    parsed = [n for n in parsed if n >= 100000]

    if len(parsed) >= 2:
        return min(parsed), max(parsed)
    elif len(parsed) == 1:
        return parsed[0], parsed[0]
    return None, None


class DatabaseManager:
    """Manages SQLite/MySQL database connections and queries."""

    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or config.DATABASE_URL or config._get_config("DATABASE_URL", "")
        
        # Format URL for MySQL and add SSL if necessary
        if self.db_url.startswith("mysql"):
            # Ensure pymysql driver is used
            if not self.db_url.startswith("mysql+pymysql://"):
                self.db_url = self.db_url.replace("mysql://", "mysql+pymysql://")
            
            # Remove any query parameters (like ?ssl-mode=REQUIRED) which can break PyMySQL
            if "?" in self.db_url:
                self.db_url = self.db_url.split("?")[0]
            
            # Aiven MySQL requires SSL
            # Check if ca.pem exists in either aiven/ca.pem or data/ca.pem
            import os
            ca_path = "aiven/ca.pem"
            if not os.path.exists(ca_path):
                ca_path = "data/ca.pem"
                
            ssl_args = {"ssl": {"ssl_ca": ca_path}}
            self.engine = create_engine(self.db_url, connect_args=ssl_args, echo=False)
        else:
            self.engine = create_engine(self.db_url, echo=False)
            
        self.Session = sessionmaker(bind=self.engine)

    def create_tables(self):
        """Create all tables."""
        Base.metadata.create_all(self.engine)

    def get_job_count(self) -> int:
        """Get total number of jobs."""
        session = self.Session()
        try:
            return session.query(Job).count()
        finally:
            session.close()

    def execute_raw_sql(self, query_str: str) -> list[dict]:
        """Execute a raw SQL query and return results as dicts. For SQL Agent."""
        session = self.Session()
        try:
            result = session.execute(sql_text(query_str))
            columns = result.keys()
            rows = result.fetchall()
            return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            return [{"error": str(e)}]
        finally:
            session.close()

