from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
import asyncio

class AsyncPlaywrightBrowser:
    def __init__(self, page_wait=20, min_text_length=500, skip_words=None, n_contexts=5, max_task_time=45):
        self.page_wait = page_wait * 1000  # ms
        self.min_text_length = min_text_length
        self.skip_words = skip_words or ['blocked', 'captcha', 'consent']
        self.n_contexts = n_contexts
        self.max_task_time = max_task_time  # seconds for killing long tasks

        self.playwright = None
        self.browser = None
        self.contexts = []

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
        if not url:
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
        """Scrape full text from a page, automatically following redirects, with a timeout."""
        if context_id < 0 or context_id >= len(self.contexts):
            raise ValueError(f"context_id must be 0–{len(self.contexts)-1}")

        ctx = self.contexts[context_id]
        page = await ctx.new_page()
        page.set_default_navigation_timeout(self.page_wait)

        try:
            final_url = await asyncio.wait_for(self.resolve_final_url(page, url), timeout=self.max_task_time)
            print(f"[AsyncBrowser] Final URL resolved: {final_url}")

            paragraphs = await asyncio.wait_for(page.query_selector_all("p, div"), timeout=self.max_task_time)
            text_blocks = []
            for p in paragraphs:
                t = (await p.text_content() or "").strip()
                if t and len(t) > 50:
                    text_blocks.append(t)

            page_text = "\n".join(text_blocks)
            await page.close()

            if len(page_text) < self.min_text_length:
                return None
            if any(word.lower() in page_text[:500].lower() for word in self.skip_words):
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
