/**
 * Enhanced JavaScript for Environmental Catastrophe Detection System
 */

// Document ready handler
document.addEventListener('DOMContentLoaded', function () {
    // Initialize tooltips if Bootstrap is available
    if (typeof bootstrap !== 'undefined') {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });

        // Initialize popovers
        const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
        popoverTriggerList.map(function (popoverTriggerEl) {
            return new bootstrap.Popover(popoverTriggerEl);
        });
    }

    // Format dates in the document
    formatDatesInDocument();

    // Handle map resize events
    window.addEventListener('resize', function () {
        resizeMap();
    });

    // Call resizeMap after page load for responsive maps
    setTimeout(resizeMap, 500);
});

/**
 * Resize map container if exists
 * The Folium maps sometimes need manual resize handling
 */
function resizeMap() {
    // Look for the Leaflet map
    const mapContainer = document.querySelector('#map-container .leaflet-container');
    if (mapContainer && typeof L !== 'undefined') {
        // Each Leaflet map instance
        const maps = Object.values(L.Map._instances || {});
        maps.forEach(map => {
            map.invalidateSize();
        });
    }
}

/**
 * Format dates in the document for better readability
 */
function formatDatesInDocument() {
    document.querySelectorAll('.format-date').forEach(function (element) {
        const dateStr = element.textContent || element.innerText;
        if (dateStr && dateStr !== 'N/A') {
            element.textContent = formatDate(dateStr);
        }
    });
}

/**
 * Format date string for display
 * 
 * @param {string} dateString - The date string to format
 * @returns {string} - Formatted date string
 */
function formatDate(dateString) {
    if (!dateString) return 'N/A';

    const date = new Date(dateString);
    if (isNaN(date.getTime())) return dateString;

    return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

/**
 * Show loading overlay
 * 
 * @param {boolean} show - Whether to show or hide the loading overlay
 */
function showLoading(show = true) {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        overlay.style.display = show ? 'flex' : 'none';
    }
}

/**
 * Show a toast notification
 * 
 * @param {string} message - The message to display
 * @param {string} type - The type of toast (success, error, warning, info)
 */
function showToast(message, type = 'info') {
    // Check if toastContainer exists, create if not
    let toastContainer = document.getElementById('toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.id = 'toast-container';
        toastContainer.className = 'position-fixed bottom-0 end-0 p-3';
        toastContainer.style.zIndex = '5';
        document.body.appendChild(toastContainer);
    }

    // Create toast element
    const toastId = 'toast-' + Date.now();
    const toast = document.createElement('div');
    toast.id = toastId;
    toast.className = `toast show bg-${type === 'error' ? 'danger' : type}`;
    toast.setAttribute('role', 'alert');
    toast.setAttribute('aria-live', 'assertive');
    toast.setAttribute('aria-atomic', 'true');

    // Toast header
    const header = document.createElement('div');
    header.className = 'toast-header';

    const title = document.createElement('strong');
    title.className = 'me-auto';
    title.textContent = type.charAt(0).toUpperCase() + type.slice(1);

    const closeButton = document.createElement('button');
    closeButton.type = 'button';
    closeButton.className = 'btn-close';
    closeButton.setAttribute('data-bs-dismiss', 'toast');
    closeButton.setAttribute('aria-label', 'Close');
    closeButton.onclick = function () {
        document.getElementById(toastId).remove();
    };

    header.appendChild(title);
    header.appendChild(closeButton);

    // Toast body
    const body = document.createElement('div');
    body.className = 'toast-body text-white';
    body.textContent = message;

    // Assemble toast
    toast.appendChild(header);
    toast.appendChild(body);

    // Add to container
    toastContainer.appendChild(toast);

    // Remove after 5 seconds
    setTimeout(function () {
        const toastElement = document.getElementById(toastId);
        if (toastElement) {
            toastElement.remove();
        }
    }, 5000);
}

/**
 * Toggle dark/light mode (if implemented)
 */
function toggleTheme() {
    // This function can be expanded if dark mode is implemented
    document.body.classList.toggle('dark-mode');
}