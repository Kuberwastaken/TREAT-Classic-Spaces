# script_search_api.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
from bs4 import BeautifulSoup
from model.analyzer import analyze_content
import logging
from difflib import get_close_matches
import re
from typing import Dict
from dataclasses import dataclass
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_URL = "https://imsdb.com"
ALL_SCRIPTS_URL = f"{BASE_URL}/all-scripts.html"

@dataclass
class ProgressInfo:
    progress: float
    status: str
    timestamp: datetime

progress_tracker: Dict[str, ProgressInfo] = {}

def update_progress(movie_name: str, progress: float, message: str):
    """
    Update the progress tracker with current progress and status message.
    """
    progress_tracker[movie_name] = ProgressInfo(
        progress=progress,
        status=message,
        timestamp=datetime.now()
    )
    logger.info(f"{message} (Progress: {progress * 100:.0f}%)")

def find_movie_link(movie_name: str, soup: BeautifulSoup) -> str | None:
    """
    Find the closest matching movie link from the script database.
    """
    movie_links = {link.text.strip().lower(): link['href'] for link in soup.find_all('a', href=True)}
    close_matches = get_close_matches(movie_name.lower(), movie_links.keys(), n=1, cutoff=0.6)
    
    if close_matches:
        logger.info(f"Close match found: {close_matches[0]}")
        return BASE_URL + movie_links[close_matches[0]]
    
    logger.info("No close match found.")
    return None

def find_script_link(soup: BeautifulSoup, movie_name: str) -> str | None:
    """
    Find the script download link for a given movie.
    """
    patterns = [
        f'Read "{movie_name}" Script',
        f'Read "{movie_name.title()}" Script',
        f'Read "{movie_name.upper()}" Script',
        f'Read "{movie_name.lower()}" Script'
    ]
    
    for link in soup.find_all('a', href=True):
        link_text = link.text.strip()
        if any(pattern.lower() in link_text.lower() for pattern in patterns):
            return link['href']
        elif all(word.lower() in link_text.lower() for word in ["Read", "Script", movie_name]):
            return link['href']
    return None

def fetch_script(movie_name: str) -> str | None:
    """
    Fetch and extract the script content for a given movie.
    """
    # Initial page load
    update_progress(movie_name, 0.1, "Fetching the script database...")
    try:
        response = requests.get(ALL_SCRIPTS_URL)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to load the main page: {str(e)}")
        return None

    # Search for movie
    update_progress(movie_name, 0.2, "Searching for the movie...")
    soup = BeautifulSoup(response.text, 'html.parser')
    movie_link = find_movie_link(movie_name, soup)
    
    if not movie_link:
        logger.error(f"Script for '{movie_name}' not found.")
        return None

    # Fetch movie page
    update_progress(movie_name, 0.3, "Loading movie details...")
    try:
        response = requests.get(movie_link)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to load the movie page: {str(e)}")
        return None

    # Find script link
    update_progress(movie_name, 0.4, "Locating script download...")
    soup = BeautifulSoup(response.text, 'html.parser')
    script_link = find_script_link(soup, movie_name)

    if not script_link:
        logger.error(f"Unable to find script link for '{movie_name}'.")
        return None

    # Fetch script content
    script_page_url = BASE_URL + script_link
    update_progress(movie_name, 0.5, "Downloading script content...")
    
    try:
        response = requests.get(script_page_url)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to load the script: {str(e)}")
        return None

    # Extract script text
    update_progress(movie_name, 0.6, "Extracting script text...")
    soup = BeautifulSoup(response.text, 'html.parser')
    script_content = soup.find('pre')
    
    if script_content:
        update_progress(movie_name, 0.7, "Script extracted successfully")
        return script_content.get_text()
    else:
        logger.error("Failed to extract script content.")
        return None

@app.get("/api/fetch_and_analyze")
async def fetch_and_analyze(movie_name: str):
    """
    Fetch and analyze a movie script, with progress tracking.
    """
    try:
        # Initialize progress
        update_progress(movie_name, 0.0, "Starting script search...")
        
        # Fetch script
        script_text = fetch_script(movie_name)
        if not script_text:
            raise HTTPException(status_code=404, detail="Script not found or error occurred")

        # Analyze content
        update_progress(movie_name, 0.8, "Analyzing script content...")
        result = await analyze_content(script_text)
        
        # Finalize
        update_progress(movie_name, 1.0, "Analysis complete!")
        return result
        
    except Exception as e:
        logger.error(f"Error in fetch_and_analyze: {str(e)}", exc_info=True)
        # Clean up progress tracker in case of error
        if movie_name in progress_tracker:
            del progress_tracker[movie_name]
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/api/progress")
def get_progress(movie_name: str):
    """
    Get the current progress and status for a movie analysis.
    """
    if movie_name not in progress_tracker:
        return {
            "progress": 0,
            "status": "Waiting to start..."
        }
    
    progress_info = progress_tracker[movie_name]
    
    # Clean up old entries (optional)
    current_time = datetime.now()
    if (current_time - progress_info.timestamp).total_seconds() > 3600:  # 1 hour timeout
        del progress_tracker[movie_name]
        return {
            "progress": 0,
            "status": "Session expired. Please try again."
        }
    
    return {
        "progress": progress_info.progress,
        "status": progress_info.status
    }

@app.on_event("startup")
async def startup_event():
    """
    Initialize the server and clear any existing progress data.
    """
    progress_tracker.clear()
    logger.info("Server started, progress tracker initialized")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)