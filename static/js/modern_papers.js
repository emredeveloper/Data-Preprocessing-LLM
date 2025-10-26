document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const papersGrid = document.getElementById('papersGrid');
    const papersList = document.getElementById('papersList');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const noResults = document.getElementById('noResults');
    const searchInput = document.getElementById('searchInput');
    const searchButton = document.getElementById('searchButton');
    const resetFiltersButton = document.getElementById('resetFilters');
    const clearFiltersBtn = document.getElementById('clearFiltersBtn');
    const categoryFilters = document.getElementById('categoryFilters');
    const dateFilter = document.getElementById('dateFilter');
    const hasCodeToggle = document.getElementById('hasCodeToggle');
    const viewOptions = document.querySelectorAll('.view-option');
    const mobileMenuBtn = document.querySelector('.btn-mobile-menu');
    const mobileCloseBtn = document.querySelector('.btn-mobile-close');
    const sidebar = document.querySelector('.sidebar');
    
    // Modal elements
    const paperModal = document.getElementById('paperModal');
    const modalBody = document.getElementById('modalBody');
    const arxivLink = document.getElementById('arxivLink');
    const pdfLink = document.getElementById('pdfLink');
    
    // State
    let allPapers = [];
    let filteredPapers = [];
    let categories = new Set();
    let currentView = 'card';
    let activeFilters = {
        search: '',
        categories: [],
        dateRange: 'all',
        hasCode: false
    };
    
    // Initialize Bootstrap modal
    const modal = new bootstrap.Modal(paperModal);
    
    // Fetch papers
    fetchPapers();
    
    // Event listeners
    searchButton.addEventListener('click', function() {
        activeFilters.search = searchInput.value.trim();
        applyFilters();
    });
    
    searchInput.addEventListener('keyup', function(e) {
        if (e.key === 'Enter') {
            activeFilters.search = searchInput.value.trim();
            applyFilters();
        }
    });
    
    dateFilter.addEventListener('change', function() {
        activeFilters.dateRange = this.value;
        applyFilters();
    });
    
    hasCodeToggle.addEventListener('change', function() {
        activeFilters.hasCode = this.checked;
        applyFilters();
    });
    
    resetFiltersButton.addEventListener('click', resetFilters);
    clearFiltersBtn.addEventListener('click', resetFilters);
    
    viewOptions.forEach(option => {
        option.addEventListener('click', function() {
            const view = this.getAttribute('data-view');
            switchView(view);
        });
    });
    
    mobileMenuBtn.addEventListener('click', function() {
        sidebar.classList.add('active');
    });
    
    mobileCloseBtn.addEventListener('click', function() {
        sidebar.classList.remove('active');
    });
    
    // Functions
    function fetchPapers() {
        loadingIndicator.classList.remove('d-none');
        papersGrid.classList.add('d-none');
        papersList.classList.add('d-none');
        noResults.classList.add('d-none');
        
        fetch('/api/papers')
            .then(async response => {
                const data = await response.json().catch(() => null);
                if (!response.ok || !data || !data.success) {
                    const errorDetails = data && data.error ? data.error : null;
                    const messageParts = [];
                    if (errorDetails) {
                        if (typeof errorDetails === 'string') {
                            messageParts.push(errorDetails);
                        } else {
                            if (errorDetails.message) messageParts.push(errorDetails.message);
                            if (errorDetails.stage) messageParts.push(`(${errorDetails.stage.split('_').join(' ')})`);
                        }
                    }
                    const message = messageParts.length > 0
                        ? messageParts.join(' ')
                        : 'Failed to load papers.';
                    throw new Error(message);
                }
                return data.papers;
            })
            .then(papers => {
                allPapers = papers;
                filteredPapers = [...allPapers];

                // Extract unique categories
                allPapers.forEach(paper => {
                    if (paper.category) {
                        paper.category.split(',').forEach(cat => {
                            categories.add(cat.trim());
                        });
                    }
                });
                
                populateCategoryFilters();
                renderPapers();
            })
            .catch(error => {
                console.error('Error fetching papers:', error);
                loadingIndicator.classList.add('d-none');
                noResults.querySelector('h3').textContent = 'Error Loading Papers';
                noResults.querySelector('p').textContent = error.message || 'Please try again later.';
                noResults.classList.remove('d-none');
            });
    }
    
    function populateCategoryFilters() {
        categoryFilters.innerHTML = '';
        
        Array.from(categories).sort().forEach(category => {
            const label = document.createElement('label');
            label.className = 'custom-checkbox';
            label.innerHTML = `
                ${category}
                <input type="checkbox" data-category="${category}">
                <span class="checkmark"></span>
            `;
            categoryFilters.appendChild(label);
        });
        
        // Add event listeners to checkboxes
        categoryFilters.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
            checkbox.addEventListener('change', function() {
                const category = this.getAttribute('data-category');
                
                if (this.checked) {
                    activeFilters.categories.push(category);
                } else {
                    activeFilters.categories = activeFilters.categories.filter(c => c !== category);
                }
                
                applyFilters();
            });
        });
    }
    
    function renderPapers() {
        loadingIndicator.classList.add('d-none');
        
        if (filteredPapers.length === 0) {
            noResults.classList.remove('d-none');
            papersGrid.classList.add('d-none');
            papersList.classList.add('d-none');
            return;
        }
        
        noResults.classList.add('d-none');
        
        // Render grid view
        papersGrid.innerHTML = '';
        filteredPapers.forEach(paper => {
            const tags = paper.tag_keywords.slice(0, 3).map(tag => 
                `<span class="paper-card-tag">${tag}</span>`
            ).join('');
            
            // Generate dynamic card style using the paper's UI theme
            const cardStyle = paper.ui_theme ? 
                `style="--accent: ${paper.ui_theme.accent};"` : '';
            
            const card = document.createElement('div');
            card.className = 'paper-card';
            card.setAttribute('data-id', paper.id);
            card.innerHTML = `
                <div class="paper-card-header" ${cardStyle}>
                    <img class="paper-card-thumb" src="${paper.thumbnail || '/static/images/papers/generic_paper.jpg'}" alt="${paper.title}">
                    <div class="paper-card-overlay"></div>
                    <span class="paper-card-category">${paper.category.split(',')[0].trim()}</span>
                </div>
                <div class="paper-card-body">
                    <h3 class="paper-card-title">${paper.title}</h3>
                    <p class="paper-card-authors">${paper.authors}</p>
                    <p class="paper-card-description">${paper.description}</p>
                    <div class="paper-card-footer">
                        <div class="paper-card-tags">
                            ${tags}
                        </div>
                        <div class="paper-card-stats">
                            <span class="paper-stat"><i class="bi bi-clock"></i>${paper.reading_time}</span>
                            <span class="paper-stat"><i class="bi bi-quote"></i>${paper.citation_count}</span>
                        </div>
                    </div>
                </div>
            `;
            papersGrid.appendChild(card);
            
            // Add click event to open modal
            card.addEventListener('click', function() {
                openPaperModal(paper);
            });
        });
        
        // Render list view
        papersList.innerHTML = '';
        filteredPapers.forEach(paper => {
            const listItem = document.createElement('div');
            listItem.className = 'paper-list-item';
            listItem.setAttribute('data-id', paper.id);
            listItem.innerHTML = `
                <img class="paper-list-thumb" src="${paper.thumbnail || '/static/images/papers/generic_paper.jpg'}" alt="${paper.title}">
                <div class="paper-list-content">
                    <h3 class="paper-list-title">${paper.title}</h3>
                    <p class="paper-list-authors">${paper.authors}</p>
                    <p class="paper-list-description">${paper.description}</p>
                </div>
                <div class="paper-list-info">
                    <span class="paper-list-category">${paper.category.split(',')[0].trim()}</span>
                    <span class="paper-list-date">${paper.pub_date}</span>
                </div>
            `;
            papersList.appendChild(listItem);
            
            // Add click event to open modal
            listItem.addEventListener('click', function() {
                openPaperModal(paper);
            });
        });
        
        // Show the current view
        if (currentView === 'card') {
            papersGrid.classList.remove('d-none');
            papersList.classList.add('d-none');
        } else {
            papersGrid.classList.add('d-none');
            papersList.classList.remove('d-none');
        }
    }
    
    function openPaperModal(paper) {
        // Set modal content
        modalBody.innerHTML = `
            <div class="paper-modal-content">
                <span class="paper-modal-category">${paper.category.split(',')[0].trim()}</span>
                <h2 class="paper-modal-title">${paper.title}</h2>
                <p class="paper-modal-authors">${paper.authors}</p>
                
                <div class="paper-modal-abstract">
                    <h4>Abstract</h4>
                    <p>${paper.abstract !== "No abstract found" ? paper.abstract : "No abstract available for this paper."}</p>
                </div>
                
                <div class="paper-modal-info">
                    <div class="info-item">
                        <div class="info-item-icon"><i class="bi bi-calendar-event"></i></div>
                        <div class="info-item-label">Published</div>
                        <div class="info-item-value">${paper.pub_date}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-item-icon"><i class="bi bi-clock"></i></div>
                        <div class="info-item-label">Reading Time</div>
                        <div class="info-item-value">${paper.reading_time}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-item-icon"><i class="bi bi-quote"></i></div>
                        <div class="info-item-label">Citations</div>
                        <div class="info-item-value">${paper.citation_count}</div>
                    </div>
                </div>
                
                ${paper.has_code ? `
                <div class="alert alert-success">
                    <i class="bi bi-code-square me-2"></i> This paper likely has code available
                </div>` : ''}
            </div>
        `;
        
        // Set links
        arxivLink.href = paper.arxiv_link;
        pdfLink.href = paper.pdf_link;
        
        // Show modal
        modal.show();
    }
    
    function applyFilters() {
        filteredPapers = allPapers.filter(paper => {
            // Search filter
            if (activeFilters.search && !paperMatchesSearch(paper, activeFilters.search)) {
                return false;
            }
            
            // Category filter
            if (activeFilters.categories.length > 0 && !paperMatchesCategories(paper, activeFilters.categories)) {
                return false;
            }
            
            // Date filter
            if (activeFilters.dateRange !== 'all' && !paperMatchesDateRange(paper, activeFilters.dateRange)) {
                return false;
            }
            
            // Has code filter
            if (activeFilters.hasCode && !paper.has_code) {
                return false;
            }
            
            return true;
        });
        
        renderPapers();
    }
    
    function paperMatchesSearch(paper, query) {
        query = query.toLowerCase();
        return (
            paper.title.toLowerCase().includes(query) ||
            paper.authors.toLowerCase().includes(query) ||
            paper.description.toLowerCase().includes(query) ||
            (paper.abstract && paper.abstract.toLowerCase().includes(query))
        );
    }
    
    function paperMatchesCategories(paper, selectedCategories) {
        if (!paper.category) return false;
        
        const paperCategories = paper.category.split(',').map(cat => cat.trim());
        return selectedCategories.some(cat => paperCategories.includes(cat));
    }
    
    function paperMatchesDateRange(paper, range) {
        if (!paper.pub_date) return true;
        
        const pubDate = new Date(paper.pub_date);
        const now = new Date();
        
        switch (range) {
            case 'today':
                return pubDate.toDateString() === now.toDateString();
            case 'week':
                const weekAgo = new Date();
                weekAgo.setDate(now.getDate() - 7);
                return pubDate >= weekAgo;
            case 'month':
                const monthAgo = new Date();
                monthAgo.setMonth(now.getMonth() - 1);
                return pubDate >= monthAgo;
            default:
                return true;
        }
    }
    
    function switchView(view) {
        currentView = view;
        
        // Update active class on view options
        viewOptions.forEach(option => {
            if (option.getAttribute('data-view') === view) {
                option.classList.add('active');
            } else {
                option.classList.remove('active');
            }
        });
        
        // Show appropriate view
        if (view === 'card') {
            papersGrid.classList.remove('d-none');
            papersList.classList.add('d-none');
        } else {
            papersGrid.classList.add('d-none');
            papersList.classList.remove('d-none');
        }
    }
    
    function resetFilters() {
        // Reset search
        searchInput.value = '';
        activeFilters.search = '';
        
        // Reset categories
        categoryFilters.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
            checkbox.checked = false;
        });
        activeFilters.categories = [];
        
        // Reset date filter
        dateFilter.value = 'all';
        activeFilters.dateRange = 'all';
        
        // Reset has code toggle
        hasCodeToggle.checked = false;
        activeFilters.hasCode = false;
        
        // Apply reset filters
        filteredPapers = [...allPapers];
        renderPapers();
    }
});
