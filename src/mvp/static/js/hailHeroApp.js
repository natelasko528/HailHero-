export function hailHeroApp() {
    return {
        leads: [],
        selectedLead: null,
        loading: true,
        viewMode: 'grid',
        mobileMenuOpen: false,
        map: null,
        markers: [],
        stats: {
            totalLeads: 0,
            activeInspections: 0,
            avgScore: 0,
            conversionRate: 0
        },
        filters: {
            search: '',
            status: '',
            sortBy: 'score'
        },
        notification: {
            show: false,
            message: '',
            type: 'info'
        },

        init() {
            this.loadLeads();
            this.initMap();
        },

        async loadLeads() {
            this.loading = true;
            try {
                const response = await fetch('/api/leads');
                this.leads = await response.json();
                this.updateStats();
                this.updateMap();
            } catch (error) {
                this.showNotification('Error loading leads', 'error');
                console.error('Error loading leads:', error);
            } finally {
                this.loading = false;
            }
        },

        updateStats() {
            this.stats.totalLeads = this.leads.length;
            this.stats.activeInspections = this.leads.filter(lead => lead.status === 'inspected').length;
            
            const totalScore = this.leads.reduce((sum, lead) => sum + (lead.score || 0), 0);
            this.stats.avgScore = this.leads.length > 0 ? (totalScore / this.leads.length).toFixed(1) : 0;
            
            this.stats.conversionRate = this.stats.totalLeads > 0 ? 
                Math.round((this.stats.activeInspections / this.stats.totalLeads) * 100) : 0;
        },

        initMap() {
            setTimeout(() => {
                if (typeof L !== 'undefined') {
                    this.map = L.map('map').setView([39.8283, -98.5795], 4);
                    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                        attribution: '© OpenStreetMap contributors'
                    }).addTo(this.map);
                    this.updateMap();
                }
            }, 100);
        },

        updateMap() {
            if (!this.map) return;
            
            // Clear existing markers
            this.markers.forEach(marker => this.map.removeLayer(marker));
            this.markers = [];
            
            // Add markers for leads
            this.leads.forEach(lead => {
                if (lead.property?.lat && lead.property?.lon) {
                    const marker = L.marker([lead.property.lat, lead.property.lon])
                        .addTo(this.map)
                        .bindPopup(`
                            <div class="p-2">
                                <h3 class="font-semibold">${lead.lead_id}</h3>
                                <p class="text-sm">Score: ${lead.score}</p>
                                <p class="text-sm">Status: ${lead.status}</p>
                            </div>
                        `);
                    this.markers.push(marker);
                }
            });
            
            // Fit map to show all markers
            if (this.markers.length > 0) {
                const group = new L.featureGroup(this.markers);
                this.map.fitBounds(group.getBounds().pad(0.1));
            }
        },

        get filteredLeads() {
            let filtered = this.leads;
            
            if (this.filters.search) {
                filtered = filtered.filter(lead => 
                    lead.lead_id.toLowerCase().includes(this.filters.search.toLowerCase()) ||
                    (lead.property?.lat && lead.property?.lon && 
                     `${lead.property.lat}, ${lead.property.lon}`.includes(this.filters.search))
                );
            }
            
            if (this.filters.status) {
                filtered = filtered.filter(lead => lead.status === this.filters.status);
            }
            
            // Sort leads
            filtered.sort((a, b) => {
                switch (this.filters.sortBy) {
                    case 'score':
                        return (b.score || 0) - (a.score || 0);
                    case 'date':
                        return new Date(b.created_ts || 0) - new Date(a.created_ts || 0);
                    case 'magnitude':
                        return (b.event?.MAGNITUDE || 0) - (a.event?.MAGNITUDE || 0);
                    default:
                        return 0;
                }
            });
            
            return filtered;
        },

        selectLead(lead) {
            this.selectedLead = lead;
            this.showNotification(`Selected lead: ${lead.lead_id}`, 'info');
        },

        showOnMap(lead) {
            if (lead.property?.lat && lead.property?.lon && this.map) {
                this.map.setView([lead.property.lat, lead.property.lon], 15);
                this.selectLead(lead);
            }
        },

        async markInspected(lead) {
            try {
                const response = await fetch('/api/inspection', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        lead_id: lead.lead_id,
                        notes: 'Inspected via modern UI',
                        photos: []
                    })
                });
                
                if (response.ok) {
                    this.showNotification('Lead marked as inspected', 'success');
                    this.loadLeads(); // Refresh data
                } else {
                    this.showNotification('Error marking lead as inspected', 'error');
                }
            } catch (error) {
                this.showNotification('Error marking lead as inspected', 'error');
                console.error('Error marking lead as inspected:', error);
            }
        },

        callLead(lead) {
            this.showNotification('Call functionality would be implemented here', 'info');
        },

        refreshData() {
            this.loadLeads();
            this.showNotification('Data refreshed', 'success');
        },

        showNotification(message, type = 'info') {
            this.notification.message = message;
            this.notification.type = type;
            this.notification.show = true;
            
            setTimeout(() => {
                this.notification.show = false;
            }, 3000);
        },

        getScoreBadgeClass(score) {
            if (score >= 20) return 'bg-red-100 text-red-800';
            if (score >= 15) return 'bg-yellow-100 text-yellow-800';
            return 'bg-green-100 text-green-800';
        },

        getStatusBadgeClass(status) {
            switch (status) {
                case 'new': return 'bg-blue-100 text-blue-800';
                case 'inspected': return 'bg-green-100 text-green-800';
                case 'qualified': return 'bg-purple-100 text-purple-800';
                default: return 'bg-gray-100 text-gray-800';
            }
        },

        getNotificationClass(type) {
            switch (type) {
                case 'success': return 'border-green-500';
                case 'error': return 'border-red-500';
                case 'warning': return 'border-yellow-500';
                default: return 'border-blue-500';
            }
        },

        getNotificationIcon(type) {
            switch (type) {
                case 'success': return 'fa-check-circle text-green-500';
                case 'error': return 'fa-exclamation-circle text-red-500';
                case 'warning': return 'fa-exclamation-triangle text-yellow-500';
                default: return 'fa-info-circle text-blue-500';
            }
        },

        formatLocation(property) {
            if (!property) return 'N/A';
            const lat = property.lat !== undefined ? property.lat.toFixed(4) : '0.0000';
            const lon = property.lon !== undefined ? property.lon.toFixed(4) : '0.0000';
            return `${lat}, ${lon}`;
        },

        formatDate(dateString) {
            if (!dateString) return 'N/A';
            return new Date(dateString).toLocaleDateString();
        }
    }
}