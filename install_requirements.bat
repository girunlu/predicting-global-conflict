@echo off
echo Installing required Python packages...

:: Upgrade pip
pip install --upgrade pip

:: Core packages
pip install openai python-dotenv gnews python-dateutil requests pandas numpy rapidfuzz

:: Newspaper3k and parsing dependencies
pip install newspaper3k lxml html5lib beautifulsoup4

:: Playwright (for JS-rendered pages)
pip install playwright
playwright install

echo ✅ All packages installed successfully.
pause
