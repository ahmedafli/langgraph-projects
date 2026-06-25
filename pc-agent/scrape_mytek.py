import httpx
from bs4 import BeautifulSoup
import json

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html",
}

def scrape_mytek_gaming_laptops():
    all_products = []
    
    # Scrape all 5 pages
    for page in range(1, 6):
        url = f"https://www.mytek.tn/informatique/ordinateurs-portables/pc-gamer.html?categoryId=40&p={page}"
        print(f"📄 Scraping page {page}...")
        
        response = httpx.get(url, headers=headers, follow_redirects=True)
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Find the seo-product-data container
        seo_container = soup.find(id="seo-product-data")
        
        if not seo_container:
            print(f"  ⚠️ No products found on page {page}")
            break
            
        # Extract all products from this page
        items = seo_container.find_all("div", attrs={"data-id": True})
        print(f"  ✅ Found {len(items)} products")
        
        for item in items:
            product = {
                "id": item.get("data-id"),
                "name": item.get("data-name"),
                "price": float(item.get("data-price", 0)),
                "final_price": float(item.get("data-final-price", 0)),
                "stock": item.get("data-erpstock"),
                "url": item.get("data-url"),
                "manufacturer": item.get("data-manufacturer"),
                "description": item.get("data-description", "")[:200]
            }
            all_products.append(product)
    
    return all_products

# Test it
products = scrape_mytek_gaming_laptops()
print(f"\n🎯 Total products found: {len(products)}")
print("\nSample products:")
for p in products[:5]:
    print(f"\n📦 {p['name']}")
    print(f"💰 Price: {p['final_price']} TND")
    print(f"📊 Stock: {p['stock']}")
    print(f"🔗 URL: {p['url']}")