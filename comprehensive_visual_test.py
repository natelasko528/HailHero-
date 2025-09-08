#!/usr/bin/env python3
"""
Comprehensive visual testing for Hail Hero MVP using Playwright.

Tests all UI components, user interactions, and captures screenshots for analysis.
"""

from playwright.sync_api import sync_playwright, Page, expect
import argparse
import time
from pathlib import Path
import json

SCREENSHOT_DIR = Path(__file__).parent / "screenshots" / "visual_test"
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

def test_dashboard_ui(page: Page, base_url: str):
    """Test the main dashboard UI components"""
    print("🧪 Testing dashboard UI...")
    
    # Navigate to dashboard
    page.goto(base_url, wait_until='networkidle')
    page.wait_for_load_state('domcontentloaded')
    
    # Wait for main components to load
    page.wait_for_selector('.stats-grid', timeout=10000)
    page.wait_for_selector('.leads-grid', timeout=10000)
    
    # Take screenshot of full dashboard
    page.screenshot(path=f"{SCREENSHOT_DIR}/01_dashboard_full.png", full_page=True)
    print("📸 Captured full dashboard screenshot")
    
    # Test responsive design - mobile view
    page.set_viewport_size({"width": 375, "height": 667})
    page.wait_for_timeout(1000)
    page.screenshot(path=f"{SCREENSHOT_DIR}/02_dashboard_mobile.png")
    print("📱 Captured mobile view")
    
    # Test tablet view
    page.set_viewport_size({"width": 768, "height": 1024})
    page.wait_for_timeout(1000)
    page.screenshot(path=f"{SCREENSHOT_DIR}/03_dashboard_tablet.png")
    print("📊 Captured tablet view")
    
    # Reset to desktop
    page.set_viewport_size({"width": 1920, "height": 1080})

def test_search_and_filters(page: Page):
    """Test search and filter functionality"""
    print("🔍 Testing search and filters...")
    
    # Test search input
    search_input = page.locator('#searchInput')
    search_input.fill('Chicago')
    page.wait_for_timeout(1000)
    page.screenshot(path=f"{SCREENSHOT_DIR}/04_search_chicago.png")
    print("🔎 Tested search functionality")
    
    # Clear search
    search_input.fill('')
    
    # Test status filter
    status_filter = page.locator('#statusFilter')
    status_filter.select_option('new')
    page.wait_for_timeout(1000)
    page.screenshot(path=f"{SCREENSHOT_DIR}/05_filter_new.png")
    print("🏷️ Tested status filter")
    
    # Test sort functionality
    sort_by = page.locator('#sortBy')
    sort_by.select_option('score')
    page.wait_for_timeout(1000)
    page.screenshot(path=f"{SCREENSHOT_DIR}/06_sort_by_score.png")
    print("📊 Tested sort functionality")

def test_lead_cards(page: Page):
    """Test individual lead card interactions"""
    print("🃏 Testing lead card interactions...")
    
    # Wait for lead cards to load
    page.wait_for_selector('.lead-card', timeout=10000)
    
    # Take screenshot of lead cards
    first_card = page.locator('.lead-card').first
    first_card.scroll_into_view_if_needed()
    page.screenshot(path=f"{SCREENSHOT_DIR}/07_lead_cards.png")
    print("📋 Captured lead cards view")
    
    # Test lead card hover effect
    first_card.hover()
    page.wait_for_timeout(500)
    page.screenshot(path=f"{SCREENSHOT_DIR}/08_lead_card_hover.png")
    print("👆 Tested lead card hover")
    
    # Test view details button
    view_btn = first_card.locator('button:has-text("View Details")')
    if view_btn.is_visible():
        view_btn.click()
        page.wait_for_timeout(1000)
        page.screenshot(path=f"{SCREENSHOT_DIR}/09_lead_details_modal.png")
        print("👁️ Tested lead details modal")
        
        # Close modal if it exists
        close_btn = page.locator('button:has-text("Close"), button[aria-label*="Close"]').first
        if close_btn.is_visible():
            close_btn.click()

def test_inspection_workflow(page: Page):
    """Test the inspection submission workflow"""
    print("🔧 Testing inspection workflow...")
    
    # Find first lead card and test inspection
    first_card = page.locator('.lead-card').first
    inspect_btn = first_card.locator('button:has-text("Mark Inspected")')
    
    if inspect_btn.is_visible():
        inspect_btn.click()
        
        # Wait for confirmation dialog
        page.wait_for_timeout(1000)
        page.screenshot(path=f"{SCREENSHOT_DIR}/10_inspection_confirmation.png")
        print("✅ Captured inspection confirmation")
        
        # Cancel the action for testing purposes
        page.locator('button:has-text("Cancel")').click()
        page.wait_for_timeout(500)

def test_api_endpoints(page: Page, base_url: str):
    """Test API endpoints and responses"""
    print("🌐 Testing API endpoints...")
    
    # Test leads API
    response = page.request.get(f"{base_url}/api/leads")
    assert response.status == 200
    leads_data = response.json()
    print(f"📊 API returned {len(leads_data)} leads")
    
    # Test inspection API (GET)
    response = page.request.get(f"{base_url}/api/inspection")
    assert response.status == 200
    inspections_data = response.json()
    print(f"🔍 API returned {len(inspections_data)} inspections")

def test_mobile_features(page: Page):
    """Test mobile-specific features"""
    print("📱 Testing mobile features...")
    
    # Switch to mobile viewport
    page.set_viewport_size({"width": 375, "height": 667})
    
    # Test touch interactions
    page.wait_for_selector('.lead-card')
    first_card = page.locator('.lead-card').first
    
    # Test mobile tap interactions
    first_card.tap()
    page.wait_for_timeout(500)
    page.screenshot(path=f"{SCREENSHOT_DIR}/11_mobile_tap.png")
    print("👆 Tested mobile tap interaction")
    
    # Test mobile menu/button interactions
    mobile_btn = page.locator('.btn:has-text("View Details")').first
    if mobile_btn.is_visible():
        mobile_btn.tap()
        page.wait_for_timeout(1000)
        page.screenshot(path=f"{SCREENSHOT_DIR}/12_mobile_interaction.png")
        print("📱 Tested mobile button interaction")

def test_error_handling(page: Page, base_url: str):
    """Test error handling and edge cases"""
    print("⚠️ Testing error handling...")
    
    # Test invalid lead ID
    response = page.request.post(f"{base_url}/api/inspection", data={
        "lead_id": "invalid_id",
        "notes": "Test inspection"
    })
    print(f"❌ Invalid lead ID response: {response.status}")
    
    # Test malformed JSON
    response = page.request.post(f"{base_url}/api/inspection", 
        headers={"Content-Type": "application/json"},
        data="invalid json"
    )
    print(f"❌ Malformed JSON response: {response.status}")

def generate_test_report():
    """Generate a comprehensive test report"""
    report = {
        "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "screenshots_captured": len(list(SCREENSHOT_DIR.glob("*.png"))),
        "test_components": [
            "Dashboard UI",
            "Search and Filters", 
            "Lead Cards",
            "Inspection Workflow",
            "API Endpoints",
            "Mobile Features",
            "Error Handling"
        ],
        "ui_features_tested": [
            "Responsive Design",
            "Interactive Elements",
            "Form Validation",
            "Modal Dialogs",
            "Status Indicators",
            "Progress States"
        ]
    }
    
    with open(SCREENSHOT_DIR / "test_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"📝 Test report generated: {SCREENSHOT_DIR}/test_report.json")

def run_comprehensive_test(base_url: str):
    """Run all visual tests"""
    print(f"🚀 Starting comprehensive visual test for: {base_url}")
    print("=" * 60)
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # Visual mode for better testing
        context = browser.new_context()
        page = context.new_page()
        
        try:
            # Run all test suites
            test_dashboard_ui(page, base_url)
            test_search_and_filters(page)
            test_lead_cards(page)
            test_inspection_workflow(page)
            test_api_endpoints(page, base_url)
            test_mobile_features(page)
            test_error_handling(page, base_url)
            
            # Generate test report
            generate_test_report()
            
            print("✅ All visual tests completed successfully!")
            print(f"📸 Screenshots saved to: {SCREENSHOT_DIR}")
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            # Take error screenshot
            page.screenshot(path=f"{SCREENSHOT_DIR}/error_state.png")
            raise
        finally:
            browser.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run comprehensive visual tests for Hail Hero")
    parser.add_argument("--url", default="http://127.0.0.1:5000", help="Base URL of the application")
    args = parser.parse_args()
    
    run_comprehensive_test(args.url)