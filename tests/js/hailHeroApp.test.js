import { jest } from '@jest/globals';

// Mock Alpine.js global
global.Alpine = {
    data: (name, callback) => {
        global[name] = callback;
    }
};

// Mock Leaflet
global.L = {
    map: jest.fn(() => ({
        setView: jest.fn(),
        removeLayer: jest.fn(),
        fitBounds: jest.fn(),
    })),
    tileLayer: jest.fn(() => ({
        addTo: jest.fn(),
    })),
    marker: jest.fn(() => ({
        addTo: jest.fn(() => ({
            bindPopup: jest.fn(),
        })),
    })),
    featureGroup: jest.fn(() => ({
        getBounds: jest.fn(() => ({
            pad: jest.fn(),
        })),
    })),
};

// Mock fetch API
global.fetch = jest.fn();

// Import the Alpine.js component
import { hailHeroApp } from '../../src/mvp/static/js/hailHeroApp.js';

describe('hailHeroApp', () => {
    let app;

    beforeEach(() => {
        // Reset mocks before each test
        jest.clearAllMocks();
        
        // Initialize the component
        app = hailHeroApp();
        // Mock init's internal calls to prevent side effects during initial state testing
        const originalLoadLeads = app.loadLeads;
        const originalInitMap = app.initMap;
        app.loadLeads = jest.fn();
        app.initMap = jest.fn();
        app.init();
        // Restore original functions for specific tests if needed
        app.originalLoadLeads = originalLoadLeads;
        app.originalInitMap = originalInitMap;
    });

    test('initial state is correct', () => {
        expect(app.leads).toEqual([]);
        expect(app.selectedLead).toBeNull();
        expect(app.loading).toBe(true);
        expect(app.viewMode).toBe('grid');
        expect(app.mobileMenuOpen).toBe(false);
        expect(app.map).toBeNull();
        expect(app.markers).toEqual([]);
        expect(app.stats).toEqual({
            totalLeads: 0,
            activeInspections: 0,
            avgScore: 0,
            conversionRate: 0
        });
        expect(app.filters).toEqual({
            search: '',
            status: '',
            sortBy: 'score'
        });
        expect(app.notification).toEqual({
            show: false,
            message: '',
            type: 'info'
        });
    });

    test('loadLeads fetches data and updates state', async () => {
        // Restore original loadLeads for this test
        app.loadLeads = app.originalLoadLeads;

        const mockLeads = [
            { lead_id: '1', score: 25, status: 'new', created_ts: '2023-01-01T12:00:00Z', property: { lat: 10, lon: 20 }, event: { MAGNITUDE: 3 } },
            { lead_id: '2', score: 18, status: 'inspected', created_ts: '2023-01-02T12:00:00Z', property: { lat: 30, lon: 40 }, event: { MAGNITUDE: 2 } },
        ];
        
        fetch.mockResolvedValueOnce({
            ok: true,
            json: async () => mockLeads,
        });

        await app.loadLeads();

        expect(fetch).toHaveBeenCalledWith('/api/leads');
        expect(app.leads).toEqual(mockLeads);
        expect(app.loading).toBe(false);
        expect(app.stats.totalLeads).toBe(2);
        expect(app.stats.activeInspections).toBe(1);
        expect(app.stats.avgScore).toBe('21.5');
        expect(app.stats.conversionRate).toBe(50);
    });

    test('updateStats calculates correct statistics', () => {
        app.leads = [
            { lead_id: '1', score: 25, status: 'new' },
            { lead_id: '2', score: 18, status: 'inspected' },
            { lead_id: '3', score: 30, status: 'qualified' },
        ];
        app.updateStats();

        expect(app.stats.totalLeads).toBe(3);
        expect(app.stats.activeInspections).toBe(1);
        expect(app.stats.avgScore).toBe('24.3');
        expect(app.stats.conversionRate).toBe(33); // 1/3 * 100 = 33.33 -> 33
    });

    test('filteredLeads applies search filter', () => {
        app.leads = [
            { lead_id: 'ABC-123', score: 20, status: 'new', property: { lat: 10, lon: 20 } },
            { lead_id: 'DEF-456', score: 15, status: 'inspected', property: { lat: 30, lon: 40 } },
        ];
        app.filters.search = 'abc';
        expect(app.filteredLeads).toEqual([app.leads[0]]);

        app.filters.search = '10, 20';
        expect(app.filteredLeads).toEqual([app.leads[0]]);
    });

    test('filteredLeads applies status filter', () => {
        app.leads = [
            { lead_id: '1', score: 20, status: 'new' },
            { lead_id: '2', score: 15, status: 'inspected' },
        ];
        app.filters.status = 'inspected';
        expect(app.filteredLeads).toEqual([app.leads[1]]);
    });

    test('filteredLeads applies sort by score', () => {
    test('filteredLeads applies sort by score', () => {
        app.leads = [
            { lead_id: '1', score: 10 },
            { lead_id: '2', score: 30 },
            { lead_id: '3', score: 20 },
        ];
        app.filters.sortBy = 'score';
        expect(app.filteredLeads).toEqual([
            { lead_id: '2', score: 30 },
            { lead_id: '3', score: 20 },
            { lead_id: '1', score: 10 },
        ]);
    });

    test('filteredLeads applies sort by date', () => {
    test('filteredLeads applies sort by date', () => {
        app.leads = [
            { lead_id: '1', created_ts: '2023-01-03T12:00:00Z' },
            { lead_id: '2', created_ts: '2023-01-01T12:00:00Z' },
            { lead_id: '3', created_ts: '2023-01-02T12:00:00Z' },
        ];
        app.filters.sortBy = 'date';
        expect(app.filteredLeads).toEqual([
            { lead_id: '1', created_ts: '2023-01-03T12:00:00Z' },
            { lead_id: '3', created_ts: '2023-01-02T12:00:00Z' },
            { lead_id: '2', created_ts: '2023-01-01T12:00:00Z' },
        ]);
    });

    test('filteredLeads applies sort by magnitude', () => {
    test('filteredLeads applies sort by magnitude', () => {
        app.leads = [
            { lead_id: '1', event: { MAGNITUDE: 1 } },
            { lead_id: '2', event: { MAGNITUDE: 3 } },
            { lead_id: '3', event: { MAGNITUDE: 2 } },
        ];
        app.filters.sortBy = 'magnitude';
        expect(app.filteredLeads).toEqual([
            { lead_id: '2', event: { MAGNITUDE: 3 } },
            { lead_id: '3', event: { MAGNITUDE: 2 } },
            { lead_id: '1', event: { MAGNITUDE: 1 } },
        ]);
    });

    test('selectLead sets selectedLead and shows notification', () => {
        const mockLead = { lead_id: 'TEST-1', score: 20, status: 'new' };
        app.selectLead(mockLead);
        expect(app.selectedLead).toEqual(mockLead);
        expect(app.notification.show).toBe(true);
        expect(app.notification.message).toBe('Selected lead: TEST-1');
        expect(app.notification.type).toBe('info');
    });

    test('markInspected sends POST request and refreshes leads on success', async () => {
        // Restore original loadLeads for this test
        app.loadLeads = app.originalLoadLeads;

        const mockLead = { lead_id: 'TEST-1', score: 20, status: 'new' };
        fetch.mockResolvedValueOnce({
            ok: true,
            json: async () => ({ message: 'Success' }),
        });
        fetch.mockResolvedValueOnce({
            ok: true,
            json: async () => [], // For the refreshData call
        });

        await app.markInspected(mockLead);

        expect(fetch).toHaveBeenCalledWith('/api/inspection', expect.objectContaining({
            method: 'POST',
            body: JSON.stringify({ lead_id: 'TEST-1', notes: 'Inspected via modern UI', photos: [] }),
        }));
        expect(app.notification.show).toBe(true);
        expect(app.notification.message).toBe('Lead marked as inspected');
        expect(app.notification.type).toBe('success');
        expect(fetch).toHaveBeenCalledWith('/api/leads'); // loadLeads called to refresh
    });

    test('markInspected shows error notification on API failure', async () => {
        const mockLead = { lead_id: 'TEST-1', score: 20, status: 'new' };
        fetch.mockRejectedValueOnce(new Error('API Error'));

        await app.markInspected(mockLead);

        expect(app.notification.show).toBe(true);
        expect(app.notification.message).toBe('Error marking lead as inspected');
        expect(app.notification.type).toBe('error');
    });

    test('callLead shows notification', () => {
        const mockLead = { lead_id: 'TEST-1', score: 20, status: 'new' };
        app.callLead(mockLead);
        expect(app.notification.show).toBe(true);
        expect(app.notification.message).toBe('Call functionality would be implemented here', 'info');
    });

    test('refreshData calls loadLeads and shows notification', async () => {
        // Restore original loadLeads for this test
        app.loadLeads = app.originalLoadLeads;

        fetch.mockResolvedValueOnce({
            ok: true,
            json: async () => [],
        });
        
        await app.refreshData();
        expect(fetch).toHaveBeenCalledWith('/api/leads');
        expect(app.notification.show).toBe(true);
        expect(app.notification.message).toBe('Data refreshed');
        expect(app.notification.type).toBe('success');
    });

    test('getScoreBadgeClass returns correct class for score', () => {
        expect(app.getScoreBadgeClass(25)).toBe('bg-red-100 text-red-800');
        expect(app.getScoreBadgeClass(18)).toBe('bg-yellow-100 text-yellow-800');
        expect(app.getScoreBadgeClass(10)).toBe('bg-green-100 text-green-800');
    });

    test('getStatusBadgeClass returns correct class for status', () => {
        expect(app.getStatusBadgeClass('new')).toBe('bg-blue-100 text-blue-800');
        expect(app.getStatusBadgeClass('inspected')).toBe('bg-green-100 text-green-800');
        expect(app.getStatusBadgeClass('qualified')).toBe('bg-purple-100 text-purple-800');
        expect(app.getStatusBadgeClass('unknown')).toBe('bg-gray-100 text-gray-800');
    });

    test('getNotificationClass returns correct class for notification type', () => {
        expect(app.getNotificationClass('success')).toBe('border-green-500');
        expect(app.getNotificationClass('error')).toBe('border-red-500');
        expect(app.getNotificationClass('warning')).toBe('border-yellow-500');
        expect(app.getNotificationClass('info')).toBe('border-blue-500');
    });

    test('getNotificationIcon returns correct icon for notification type', () => {
        expect(app.getNotificationIcon('success')).toBe('fa-check-circle text-green-500');
        expect(app.getNotificationIcon('error')).toBe('fa-exclamation-circle text-red-500');
        expect(app.getNotificationIcon('warning')).toBe('fa-exclamation-triangle text-yellow-500');
        expect(app.getNotificationIcon('info')).toBe('fa-info-circle text-blue-500');
    });

    test('formatLocation returns formatted string or N/A', () => {
        expect(app.formatLocation({ lat: 12.34567, lon: -78.90123 })).toBe('12.3457, -78.9012');
        expect(app.formatLocation(null)).toBe('N/A');
        expect(app.formatLocation({})).toBe('0.0000, 0.0000');
    });

    test('formatDate returns formatted date string or N/A', () => {
        expect(app.formatDate('2023-01-01T12:00:00Z')).toBe(new Date('2023-01-01T12:00:00Z').toLocaleDateString());
        expect(app.formatDate(null)).toBe('N/A');
    });
});