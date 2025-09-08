#!/usr/bin/env python3
"""
Screenshot capture script for UI/UX analysis
Captures the Flask app at different viewport sizes
"""

import asyncio
from playwright.async_api import async_playwright
import os
from datetime import datetime

async def capture_screenshots():
    """Capture screenshots at different viewport sizes"""
    
    # Viewport sizes to test
    viewports = [
        {"name": "desktop", "width": 1920, "height": 1080},
        {"name": "tablet", "width": 768, "height": 1024},
        {"name": "mobile", "width": 375, "height": 667}
    ]
    
    # Pages to capture
    pages = [
        {"name": "home", "url": "http://localhost:5000/"},
        {"name": "leads", "url": "http://localhost:5000/api/leads"},
    ]
    
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        
        for viewport in viewports:
            print(f"\nCapturing {viewport['name']} viewport ({viewport['width']}x{viewport['height']})")
            
            for page_info in pages:
                try:
                    page = await browser.new_page()
                    await page.set_viewport_size({
                        "width": viewport["width"], 
                        "height": viewport["height"]
                    })
                    
                    print(f"  - Loading {page_info['name']} page...")
                    await page.goto(page_info['url'])
                    
                    # Wait for page to load
                    await page.wait_for_load_state('networkidle')
                    
                    # Create output directory
                    output_dir = f"screenshots/{viewport['name']}"
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Capture screenshot
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{output_dir}/{page_info['name']}_{timestamp}.png"
                    
                    await page.screenshot(path=filename, full_page=True)
                    print(f"    ✓ Saved: {filename}")
                    
                    # Also capture some interactive elements
                    if page_info['name'] == 'home':
                        # Try to find and click inspection button to see workflow
                        try:
                            inspect_button = await page.wait_for_selector('button:has-text("Inspect")', timeout=5000)
                            await inspect_button.click()
                            await page.wait_for_load_state('networkidle')
                            
                            workflow_filename = f"{output_dir}/inspection_workflow_{timestamp}.png"
                            await page.screenshot(path=workflow_filename, full_page=True)
                            print(f"    ✓ Saved workflow: {workflow_filename}")
                        except:
                            print("    ! Inspection workflow not found")
                    
                    await page.close()
                    
                except Exception as e:
                    print(f"    ✗ Error capturing {page_info['name']}: {e}")
        
        await browser.close()
        print(f"\n✅ Screenshot capture complete!")

if __name__ == "__main__":
    asyncio.run(capture_screenshots())