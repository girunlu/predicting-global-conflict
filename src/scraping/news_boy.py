from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
import asyncio

class AsyncPlaywrightBrowser:
    """
    An asynchronous browser automation class using Playwright for scraping web page text with concurrency control.
    Attributes:
        page_wait (int): Time to wait for page navigation and loading, in milliseconds.
        min_text_length (int): Minimum length of text to consider a page successfully scraped.
        skip_words (list[str]): List of words indicating blocked or unwanted pages.
        n_contexts (int): Number of browser contexts to use for parallel scraping.
        max_task_time (int): Maximum time (in seconds) allowed for a single scraping task before it is killed.
        max_concurrent_tasks (int): Maximum number of concurrent scraping tasks.
        playwright: Playwright instance.
        browser: Playwright browser instance.
        contexts (list): List of browser contexts for parallel scraping.
        semaphore (asyncio.Semaphore): Semaphore to limit concurrent tasks.
    Methods:
        start():
            Asynchronously starts the Playwright browser and initializes contexts.
        end():
            Asynchronously closes all browser contexts, the browser, and stops Playwright.
        resolve_final_url(page, url):
            Asynchronously navigates to the given URL, follows redirects, and returns the final landing URL.
            Handles special cases like Google RSS redirects.
        get_page_text(url, context_id=0):
            Asynchronously scrapes visible text from the given URL using the specified browser context.
            Applies concurrency limits, minimum text length, and skip word filtering.
            Returns the scraped text or None if scraping fails or content is insufficient.
    """
    def __init__(self, page_wait=20, min_text_length=500, skip_words=None, n_contexts=5, max_task_time=45, max_concurrent_tasks = 50):
        """Initialize the AsyncPlaywrightBrowser with configuration parameters."""
        self.page_wait = page_wait * 1000  # ms
        self.min_text_length = min_text_length
        self.skip_words = skip_words or ['blocked', 'captcha', 'consent']
        self.n_contexts = n_contexts
        self.max_task_time = max_task_time  # seconds for killing long tasks
        self.max_concurrent_tasks = max_concurrent_tasks

        self.playwright = None
        self.browser = None
        self.contexts = []
        self.semaphore = asyncio.Semaphore(self.max_concurrent_tasks)

    async def start(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True, args=[
            "--disable-images",
            "--disable-background-timer-throttling",
            "--disable-renderer-backgrounding",
            "--disable-gpu",
            "--mute-audio",
            "--no-sandbox",
        ])
        self.contexts = [await self.browser.new_context() for _ in range(self.n_contexts)]

    async def end(self):
        for ctx in self.contexts:
            try:
                await ctx.close()
            except:
                pass
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    async def resolve_final_url(self, page, url: str) -> str:
        """Detects redirects (e.g., Google RSS) and waits for the final landing page."""
        if not url or "google" not in url:
            return url
        try:
            await asyncio.wait_for(page.goto(url, wait_until="domcontentloaded"), timeout=self.max_task_time)
            await asyncio.wait_for(page.wait_for_load_state("networkidle"), timeout=self.max_task_time)

            if "news.google.com/rss/articles" in page.url:
                print(f"[AsyncBrowser] Detected Google RSS redirect: {page.url}")
                try:
                    await asyncio.wait_for(page.wait_for_url("**", timeout=self.page_wait), timeout=self.max_task_time)
                except PlaywrightTimeoutError:
                    print(f"[AsyncBrowser] Timeout waiting for redirect, using current URL")

            return page.url
        except asyncio.TimeoutError:
            print(f"[AsyncBrowser] Task killed due to timeout for URL: {url}")
            return page.url
        except Exception as e:
            print(f"[AsyncBrowser] Failed to resolve final URL {url}: {e}")
            return url

    async def get_page_text(self, url: str, context_id=0) -> str | None:
        """Scrape visible, meaningful text from a URL using robust fallbacks."""
        async with self.semaphore:
            if context_id < 0 or context_id >= len(self.contexts):
                raise ValueError(f"context_id must be 0–{len(self.contexts)-1}")

            ctx = self.contexts[context_id]
            page = await ctx.new_page()
            page.set_default_navigation_timeout(self.page_wait)

            try:
                final_url = await asyncio.wait_for(
                    self.resolve_final_url(page, url),
                    timeout=self.max_task_time
                )
                print(f"[AsyncBrowser] Final URL resolved: {final_url}")

                # Wait for network + JS to settle
                await page.wait_for_load_state("networkidle")
                await asyncio.sleep(5)

                # Try multiple content selectors
                content_selectors = [
                    "article", "main", "section",
                    "div[class*='content']",
                    "div[class*='article']",
                    "div[class*='story']",
                    "div[data-component*='text']",
                    "div[class*='body']",
                    "div[class*='text']",
                    "p"
                ]

                elements = []
                for sel in content_selectors:
                    try:
                        found = await page.query_selector_all(sel)
                        elements.extend(found)
                    except Exception:
                        continue

                text_blocks = []

                # Try extracting visible elements first
                for el in elements:
                    try:
                        box = await el.bounding_box()
                        if not box:  # invisible or off-screen
                            continue
                        t = (await el.inner_text() or "").strip()
                        if not t or len(t) < 50:
                            continue
                        if len(t.split()) < len(t) / 4:  
                            continue
                        if any(word in t.lower() for word in self.skip_words):
                            continue
                        text_blocks.append(t)
                    except Exception:
                        continue

                # If still empty, fallback to full page content
                if not text_blocks:
                    print(f"[AsyncBrowser] No visible text found, falling back to raw HTML parse for {url}")
                    html = await page.content()
                    import re
                    from bs4 import BeautifulSoup

                    soup = BeautifulSoup(html, "html.parser")
                    # Get all paragraphs and texty divs
                    candidates = soup.find_all(["p", "article", "main", "section", "div"])
                    for c in candidates:
                        t = c.get_text(" ", strip=True)
                        if len(t) > 100 and not re.search(r"\b(subscribe|cookie|sign in|follow us)\b", t.lower()):
                            text_blocks.append(t)

                # Combine and clean
                page_text = "\n".join(text_blocks)
                page_text = " ".join(page_text.split())
                await page.close()

                if len(page_text) < self.min_text_length:
                    print(f"[AsyncBrowser] Text too short for {url} ({len(page_text)} chars)")
                    return None

                return page_text

            except asyncio.TimeoutError:
                print(f"[AsyncBrowser] Task killed due to timeout while scraping: {url}")
                await page.close()
                return None
            except Exception as e:
                print(f"[AsyncBrowser] Failed for {url}: {e}")
                await page.close()
                return None
            
def main():
    urls = [
        'https://www.bbc.com/news/world-asia-50883812',
        "https://news.google.com/rss/articles/CBMi6wFBVV95cUxObzFlLWVIa2tqbzUyYmdWcDNVZFJZcTZManpQU2dVNnhkVmg5S3ExUUNWLWx1Y2NVaHdwM1pJZXJjeWlWQ0huU2hPRHhCYVl3M2NmSXgyNmwtQ0lsb2M2RWpwbTRHVkpqNTRxSkcwQWV6S2RFLVVqbW5oT0RwRWxlaENhblJUUkZORGh6YVBfNWl6b2JFa3FlRUxkTjl4X0ZwTEFiRnlPb2VNeXA2NVVwdi1ZQ2NwWXFfVzdnTTVCU0lKdDBkd3hzMTNLZTZHdTAxcEw4c0hpb2JhZUwwdndXYzdhTGxHY2Y2bGVR?oc=5&hl=en-ZA&gl=ZA&ceid=ZA:en",
        "https://news.google.com/rss/articles/CBMilgFBVV95cUxPbFU1YkRGN2pEU3VYdVBDOXB3QmZzWVg4V2lJRFZQazIxTE5kbUlqUWJZcF9lV1hreW9YeEVuRnlzRGFqdVZWeGU0NUVydWZYYXNLaTNjVk5laGNvY0VCUU5FYWZSc0xwQmg2VDFtTkcyUWpXRWVXME5fN1UwbGsycHBMdFdqeXlSN0o3QnZnT3Vibl8yMmc?oc=5&hl=en-ZA&gl=ZA&ceid=ZA:en"
    ]

    asBrowser = AsyncPlaywrightBrowser()

    async def run():
        await asBrowser.start()
        # create a task per URL, round-robin assign contexts
        tasks = [
            asyncio.create_task(asBrowser.get_page_text(url, context_id=(i % asBrowser.n_contexts)))
            for i, url in enumerate(urls)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for url, res in zip(urls, results):
            if isinstance(res, Exception):
                print(f"[Main] Error scraping {url}: {res}")
            elif res:
                print(f"[Main] Success scraping {url} (first 500 chars):\n{res[:500]}\n")
            else:
                print(f"[Main] Failed to scrape or text too short for {url}")

        await asBrowser.end()

    asyncio.run(run())


if __name__ == "__main__":
    main()