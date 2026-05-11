import asyncio
import logging
import os
from typing import List, Dict
from pyppeteer import launch

logger = logging.getLogger("JulyEngine.Services.ScraperService")

class ScraperService:
    async def scrape_urls(self, urls: List[str]) -> List[Dict[str, str]]:
        results = []
        browser = None
        try:
            # Tenta encontrar o executável do Chrome/Chromium no Windows/Linux
            # Se a variável de ambiente PUPPETEER_EXECUTABLE_PATH estiver definida, ela tem prioridade
            executable_path = os.environ.get("PUPPETEER_EXECUTABLE_PATH")
            
            # Puppeteer (Pyppeteer) launch
            browser_args = ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage']
            
            logger.info("Attempting to launch Puppeteer...")
            try:
                browser = await launch(
                    headless=True,
                    executablePath=executable_path,
                    args=browser_args,
                    handleSIGINT=False,
                    handleSIGTERM=False,
                    handleSIGHUP=False
                )
            except Exception as inner_e:
                logger.warning(f"Default Puppeteer launch failed: {inner_e}. Retrying with auto-download disabled if local Chrome exists...")
                # Fallback para caminhos comuns se não encontrar o do pyppeteer
                common_paths = [
                    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                    "/usr/bin/google-chrome",
                    "/usr/bin/chromium-browser"
                ]
                for path in common_paths:
                    if os.path.exists(path):
                        logger.info(f"Found Chrome at {path}, using it.")
                        browser = await launch(
                            headless=True,
                            executablePath=path,
                            args=browser_args,
                            handleSIGINT=False,
                            handleSIGTERM=False,
                            handleSIGHUP=False
                        )
                        break
                
                if not browser:
                    raise inner_e
            
            # Scrape in parallel with a limit
            semaphore = asyncio.Semaphore(3)
            
            async def scrape_url(url: str):
                async with semaphore:
                    page = await browser.newPage()
                    # Set a real user agent
                    await page.setUserAgent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36")
                    
                    try:
                        logger.info(f"Puppeteer Scraping URL: {url}")
                        # Navigate with timeout
                        await page.goto(url, {'waitUntil': 'domcontentloaded', 'timeout': 30000})
                        
                        # Extract text content using Puppeteer evaluate
                        content = await page.evaluate("() => document.body.innerText")
                        
                        # Basic cleaning
                        content = " ".join(content.split())
                        
                        results.append({
                            "url": url,
                            "content": content[:10000] # Limit content per page
                        })
                    except Exception as e:
                        logger.error(f"Puppeteer failed to scrape {url}: {e}")
                        results.append({
                            "url": url,
                            "content": f"Error: Could not scrape this page with Puppeteer. {str(e)}"
                        })
                    finally:
                        await page.close()

            await asyncio.gather(*[scrape_url(url) for url in urls])
            
        except Exception as e:
            logger.error(f"Puppeteer Browser Error: {e}")
            results.append({
                "url": "system",
                "content": f"ERRO CRÍTICO: O Puppeteer não conseguiu iniciar. Certifique-se de que o Chrome está instalado ou execute 'pyppeteer-install' no terminal do Engine. Detalhes: {str(e)}"
            })
        finally:
            if browser:
                await browser.close()
            
        return results

scraper_service = ScraperService()

