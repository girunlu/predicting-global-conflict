from newspaper import Article
from playwright.sync_api import sync_playwright
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def get_news_site(url, page_delay = 3000, overall_timeout = 30000):
    # 1️⃣ Try newspaper3k
    try:
        article = Article(url)
        article.download()
        article.parse()
        full_text = article.text
        if full_text and full_text.strip():
            if len(full_text) > 1000:
                return full_text
        print(f"newspaper3k returned empty for {url}, trying Playwright...")
    except Exception as e:
        print(f"newspaper3k failed for {url}: {e}, trying Playwright...")

    # 2️⃣ Fallback: Playwright for JS-rendered content
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=overall_timeout)
            page.wait_for_timeout(page_delay)  # small wait for JS content

            # Grab all paragraphs and list items
            paragraphs = page.query_selector_all("p, li")
            text_blocks = [
                p.text_content().strip()
                for p in paragraphs
                if p.text_content() and len(p.text_content().strip()) > 30
            ]

            browser.close()

            full_text = "\n".join(text_blocks)
            if full_text.strip():
                return full_text
            else:
                print(f"Playwright returned empty text for {url}")
                return None
    except Exception as e:
        print(f"Playwright failed for {url}: {e}")
        return None

class BrowserSim:
    def __init__(self, page_wait=15, min_text_length=500, skip_words=None):
        self.page_wait = page_wait
        self.min_text_length = min_text_length
        self.skip_words = skip_words or ['blocked', 'captcha', 'consent']

        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-background-networking")
        options.add_argument("--disable-default-apps")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-sync")
        options.add_argument("--metrics-recording-only")
        options.add_argument("--no-first-run")
        self.options = options

    def start(self):
        self.driver = webdriver.Chrome(options=self.options)

    def get_page(self, url):
        try:
            self.driver.get(url)

            # Wait for body to exist
            WebDriverWait(self.driver, self.page_wait).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            time.sleep(1)  # let JS finish rendering

            # Try multiple selectors
            elements = self.driver.find_elements(By.TAG_NAME, "p") + \
                       self.driver.find_elements(By.TAG_NAME, "div")

            page_text = "\n".join([el.text for el in elements if len(el.text.strip()) > 50])

            # Skip short or blocked pages
            if len(page_text) < self.min_text_length:
                print(f"Page too short ({len(page_text)} chars), skipping: {url}")
                return None
            if any(word.lower() in page_text[:500].lower() for word in self.skip_words):
                print(f"Page rejected due to skip words: {url}")
                return None

            return page_text

        except Exception as e:
            print(f"Selenium failed for {url}: {e}")
            return None

    def end(self):
        self.driver.quit()
