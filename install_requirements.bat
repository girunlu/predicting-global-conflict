@echo off
echo Installing required Python packages...

:: Upgrade pip
pip install --upgrade pip

:: Core packages
pip install openai python-dotenv gnews python-dateutil requests

:: Newspaper3k and parsing dependencies
pip install newspaper3k lxml html5lib beautifulsoup4

:: NLP packages
pip install spacy nltk

:: Playwright (for JS-rendered pages)
pip install playwright
playwright install

:: Selenium and ChromeDriver
pip install selenium
:: Note: Ensure ChromeDriver matches your Chrome version
:: You may use: webdriver-manager for automatic updates
pip install webdriver-manager

:: NLTK data
python -m nltk.downloader punkt

:: Spacy English large model
python -m spacy download en_core_web_lg

echo ✅ All packages installed successfully.
pause
