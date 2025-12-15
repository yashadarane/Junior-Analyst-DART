import requests
import xml.etree.ElementTree as ET
import time
import random

def fetch_google_news_rss(ticker):
    time.sleep(random.uniform(1.5, 3.5))
    search_term = ticker.replace(".NS", "") + " stock news India"
    url = f"https://news.google.com/rss/search?q={search_term}&hl=en-IN&gl=IN&ceid=IN:en"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            headlines = []
            for item in root.findall("./channel/item")[:3]:
                title = item.find("title").text
                if "-" in title:
                    title = title.rsplit("-", 1)[0].strip()
                headlines.append(title)
            
            if headlines:
                return ". ".join(headlines)
                
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
        pass    
    return ""

def get_real_news(ticker, use_fallback=True):
    news_text = fetch_google_news_rss(ticker)
    if news_text:
        return news_text
    if use_fallback:
        return "Market volatility observed. Sector performance mixed."
    return ""