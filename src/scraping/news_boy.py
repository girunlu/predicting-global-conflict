from newspaper import Article
from playwright.sync_api import sync_playwright
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

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
    def __init__(self, page_wait = 15, min_text_length = 500, skip_words = ['blocked']):
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        self.options = options
        self.page_wait = page_wait
        self.min_text_length = min_text_length
        self.skip_words = skip_words
        
    def start(self):
        self.driver = webdriver.Chrome(options=self.options)

    def get_page(self, url):
        self.start()
        try:
            self.driver.get(url)

            # Wait until body has some text (timeout after 15s)
            WebDriverWait(self.driver, self.page_wait).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )

            p_text,div_text = "", ""
            body_text = self.driver.find_element("tag name", "body").text
            if len(body_text) < self.min_text_length:
                print("Body text too short, trying alternative selectors...")

                p_text = self.driver.find_element("tag name", "p").text
                if len(p_text) < self.min_text_length:
                    print("Paragraph text too short, trying alternative selectors...")
                    div_text = self.driver.find_element("tag name", "div").text
                    if len(p_text) < self.min_text_length:
                        print("All methods failed to get sufficient text.")
                        return None
                    else:
                        page_text = div_text
                else:
                    page_text = p_text
            else:
                page_text = body_text

            for word in self.skip_words:
                if word in page_text[:100]:
                    print(f"Website rejected request, failed for {url}...")
                    return None
            print(page_text[:100])
            return page_text
        except Exception as e:
            print(f"Continuing... but\nSelenium failed for {url}: {e}")

    def end(self):
        self.driver.quit()
        
def testing():
    url = "https://www.aljazeera.com/news/2025/8/9/india-says-six-pakistani-aircraft-shot-down-during-kashmir-conflict"
    browser = BrowserSim()
    browser.start()
    full_text = browser.get_page(url)
    browser.end()

if __name__ == "__main__":
    testing()
