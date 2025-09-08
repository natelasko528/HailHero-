#!/usr/bin/env python3
"""
Detailed UI/UX analysis script for Hail Hero Flask app
"""

import asyncio
from playwright.async_api import async_playwright
import os
from datetime import datetime

async def analyze_ui_ux():
    """Comprehensive UI/UX analysis across different viewports"""
    
    # Viewport sizes to test
    viewports = [
        {"name": "desktop", "width": 1920, "height": 1080},
        {"name": "tablet", "width": 768, "height": 1024},
        {"name": "mobile", "width": 375, "height": 667}
    ]
    
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        
        for viewport in viewports:
            print(f"\n{'='*60}")
            print(f"ANALYZING {viewport['name'].upper()} VIEWPORT")
            print(f"{'='*60}")
            print(f"Dimensions: {viewport['width']}x{viewport['height']}")
            
            page = await browser.new_page()
            await page.set_viewport_size({
                "width": viewport["width"], 
                "height": viewport["height"]
            })
            
            # Load main page
            print(f"\n1. Loading main page...")
            await page.goto("http://localhost:5000/")
            await page.wait_for_load_state('networkidle')
            
            # Capture viewport info
            viewport_info = await page.evaluate("""
                () => ({
                    screenWidth: window.screen.width,
                    screenHeight: window.screen.height,
                    innerWidth: window.innerWidth,
                    innerHeight: window.innerHeight,
                    devicePixelRatio: window.devicePixelRatio,
                    isMobile: /Mobile|Android|iPhone|iPad|iPod/i.test(navigator.userAgent)
                })
            """)
            
            print(f"   Screen: {viewport_info['screenWidth']}x{viewport_info['screenHeight']}")
            print(f"   Viewport: {viewport_info['innerWidth']}x{viewport_info['innerHeight']}")
            print(f"   Device Pixel Ratio: {viewport_info['devicePixelRatio']}")
            print(f"   Is Mobile: {viewport_info['isMobile']}")
            
            # Analyze layout
            layout_analysis = await page.evaluate("""
                () => {
                    const header = document.querySelector('header');
                    const container = document.querySelector('.container');
                    const leadsGrid = document.querySelector('.leads-grid');
                    const statCards = document.querySelectorAll('.stat-card');
                    const leadCards = document.querySelectorAll('.lead-card');
                    
                    return {
                        hasHeader: !!header,
                        headerHeight: header ? header.offsetHeight : 0,
                        containerWidth: container ? container.offsetWidth : 0,
                        leadsGridColumns: leadsGrid ? window.getComputedStyle(leadsGrid).gridTemplateColumns : 'none',
                        statCardCount: statCards.length,
                        leadCardCount: leadCards.length,
                        hasMobileMenu: !!document.querySelector('.mobile-menu-toggle'),
                        mobileMenuVisible: document.querySelector('.mobile-menu-toggle') ? 
                            window.getComputedStyle(document.querySelector('.mobile-menu-toggle')).display !== 'none' : false
                    }
                }
            """)
            
            print(f"\n2. Layout Analysis:")
            print(f"   Header present: {layout_analysis['hasHeader']}")
            print(f"   Header height: {layout_analysis['headerHeight']}px")
            print(f"   Container width: {layout_analysis['containerWidth']}px")
            print(f"   Grid columns: {layout_analysis['leadsGridColumns']}")
            print(f"   Stat cards: {layout_analysis['statCardCount']}")
            print(f"   Lead cards: {layout_analysis['leadCardCount']}")
            print(f"   Mobile menu available: {layout_analysis['hasMobileMenu']}")
            print(f"   Mobile menu visible: {layout_analysis['mobileMenuVisible']}")
            
            # Test responsive behavior
            print(f"\n3. Responsive Behavior:")
            
            # Check if grid layout adapts
            grid_adapts = await page.evaluate("""
                () => {
                    const leadsGrid = document.querySelector('.leads-grid');
                    if (!leadsGrid) return false;
                    
                    const computedStyle = window.getComputedStyle(leadsGrid);
                    const gridTemplateColumns = computedStyle.gridTemplateColumns;
                    
                    // Check if it's responsive (not fixed)
                    return gridTemplateColumns.includes('minmax') || 
                           gridTemplateColumns.includes('repeat') ||
                           gridTemplateColumns.includes('auto');
                }
            """)
            
            print(f"   Grid layout adapts: {grid_adapts}")
            
            # Check font sizes and spacing
            typography_analysis = await page.evaluate("""
                () => {
                    const body = document.body;
                    const computedStyle = window.getComputedStyle(body);
                    const leadCard = document.querySelector('.lead-card');
                    
                    return {
                        bodyFontSize: computedStyle.fontSize,
                        bodyLineHeight: computedStyle.lineHeight,
                        leadCardFontSize: leadCard ? window.getComputedStyle(leadCard).fontSize : 'N/A',
                        hasRelativeUnits: computedStyle.fontSize.includes('rem') || computedStyle.fontSize.includes('em')
                    }
                }
            """)
            
            print(f"   Body font size: {typography_analysis['bodyFontSize']}")
            print(f"   Body line height: {typography_analysis['bodyLineHeight']}")
            print(f"   Uses relative units: {typography_analysis['hasRelativeUnits']}")
            
            # Test interactive elements
            print(f"\n4. Interactive Elements:")
            
            # Check buttons and links
            interactive_analysis = await page.evaluate("""
                () => {
                    const buttons = document.querySelectorAll('button');
                    const links = document.querySelectorAll('a');
                    const inputs = document.querySelectorAll('input, select');
                    
                    return {
                        buttonCount: buttons.length,
                        linkCount: links.length,
                        inputCount: inputs.length,
                        hasTouchTargets: buttons.length > 0 || links.length > 0,
                        hasLargeTouchTargets: Array.from(buttons).some(btn => {
                            const rect = btn.getBoundingClientRect();
                            return rect.width >= 44 && rect.height >= 44; // Apple's HIG guidelines
                        })
                    }
                }
            """)
            
            print(f"   Buttons: {interactive_analysis['buttonCount']}")
            print(f"   Links: {interactive_analysis['linkCount']}")
            print(f"   Form inputs: {interactive_analysis['inputCount']}")
            print(f"   Has touch targets: {interactive_analysis['hasTouchTargets']}")
            print(f"   Has large touch targets: {interactive_analysis['hasLargeTouchTargets']}")
            
            # Test mobile-specific features
            print(f"\n5. Mobile-Specific Features:")
            
            mobile_features = await page.evaluate("""
                () => {
                    const hasViewportMeta = !!document.querySelector('meta[name="viewport"]');
                    const hasTouchIcons = !!document.querySelector('link[rel="apple-touch-icon"]');
                    const hasResponsiveImages = document.querySelectorAll('img[srcset], img[sizes]').length > 0;
                    const hasFlexibleMedia = document.querySelectorAll('img, video').length > 0;
                    
                    return {
                        hasViewportMeta: hasViewportMeta,
                        hasTouchIcons: hasTouchIcons,
                        hasResponsiveImages: hasResponsiveImages,
                        hasFlexibleMedia: hasFlexibleMedia,
                        viewportContent: hasViewportMeta ? document.querySelector('meta[name="viewport"]').content : 'none'
                    }
                }
            """)
            
            print(f"   Viewport meta tag: {mobile_features['hasViewportMeta']}")
            print(f"   Viewport content: {mobile_features['viewportContent']}")
            print(f"   Touch icons: {mobile_features['hasTouchIcons']}")
            print(f"   Responsive images: {mobile_features['hasResponsiveImages']}")
            print(f"   Flexible media: {mobile_features['hasFlexibleMedia']}")
            
            # Capture screenshots for documentation
            output_dir = f"analysis_screenshots/{viewport['name']}"
            os.makedirs(output_dir, exist_ok=True)
            
            # Full page screenshot
            full_page = f"{output_dir}/full_page_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            await page.screenshot(path=full_page, full_page=True)
            print(f"\n6. Screenshots saved:")
            print(f"   Full page: {full_page}")
            
            # Capture specific sections
            try:
                # Header section
                header = await page.query_selector('header')
                if header:
                    header_shot = f"{output_dir}/header_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    await header.screenshot(path=header_shot)
                    print(f"   Header: {header_shot}")
                
                # Stats section
                stats = await page.query_selector('.stats-dashboard')
                if stats:
                    stats_shot = f"{output_dir}/stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    await stats.screenshot(path=stats_shot)
                    print(f"   Stats dashboard: {stats_shot}")
                
                # Lead cards
                lead_card = await page.query_selector('.lead-card')
                if lead_card:
                    lead_shot = f"{output_dir}/lead_card_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    await lead_card.screenshot(path=lead_shot)
                    print(f"   Lead card: {lead_shot}")
                
            except Exception as e:
                print(f"   Error capturing section screenshots: {e}")
            
            # Test functionality
            print(f"\n7. Functionality Test:")
            
            # Test search functionality
            try:
                search_input = await page.wait_for_selector('#searchInput', timeout=5000)
                await search_input.fill('test')
                print(f"   Search input: ✓ Working")
            except:
                print(f"   Search input: ✗ Not found or not working")
            
            # Test filter functionality
            try:
                status_filter = await page.wait_for_selector('#statusFilter', timeout=5000)
                await status_filter.select_option('new')
                print(f"   Status filter: ✓ Working")
            except:
                print(f"   Status filter: ✗ Not found or not working")
            
            # Test view toggle
            try:
                list_view_btn = await page.wait_for_selector('button[onclick="setViewMode(\'list\')"]', timeout=5000)
                await list_view_btn.click()
                print(f"   List view toggle: ✓ Working")
            except:
                print(f"   List view toggle: ✗ Not found or not working")
            
            await page.close()
        
        await browser.close()
        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*60}")

if __name__ == "__main__":
    asyncio.run(analyze_ui_ux())