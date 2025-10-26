document.addEventListener('DOMContentLoaded', function() {
    // UI elements
    const papersList = document.getElementById('papersList');
    const papersLoading = document.getElementById('papersLoading');
    const noResults = document.getElementById('noResults');
    const categoryFilters = document.getElementById('categoryFilters');
    const searchInput = document.getElementById('searchInput');
    const searchButton = document.getElementById('searchButton');
    const resetFiltersButton = document.getElementById('resetFilters');
    const dateFilter = document.getElementById('dateFilter');
    
    // State
    let allPapers = [];
    let filteredPapers = [];
    let categories = new Set();
    let activeFilters = {
        search: '',
        categories: [],
        dateRange: 'all'
    };
    
    // Fetch papers
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
            
            // Populate category filters
            populateCategoryFilters();
            
            // Render papers
            renderPapers();
        })
        .catch(error => {
            console.error('Error fetching papers:', error);
            papersList.innerHTML = `
                <div class="alert alert-danger">
                    ${error.message || 'Failed to load papers. Please try again later.'}
                </div>
            `;
            papersLoading.classList.add('d-none');
            papersList.classList.remove('d-none');
        });
    
    // Populate category filters
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
    
    // Render papers
    function renderPapers() {
        papersLoading.classList.add('d-none');
        
        if (filteredPapers.length === 0) {
            noResults.classList.remove('d-none');
            papersList.classList.add('d-none');
            return;
        }
        
        noResults.classList.add('d-none');
        papersList.classList.remove('d-none');
        
        papersList.innerHTML = '';
        
        filteredPapers.forEach(paper => {
            // Format categories as badges
            const categoryBadges = paper.category 
                ? paper.category.split(',').map(cat => 
                    `<span class="category-badge">${cat.trim()}</span>`
                  ).join('')
                : '';
            
            const paperCard = document.createElement('div');
            paperCard.className = 'card paper-card shadow-sm';
            paperCard.innerHTML = `
                <div class="card-body">
                    <h5 class="card-title">
                        <a href="/paper/${paper.id}" class="text-decoration-none">${paper.title}</a>
                    </h5>
                    <p class="authors mb-2">${paper.authors}</p>
                    <div class="categories mb-2">
                        ${categoryBadges}
                    </div>
                    <p class="description">${paper.description}</p>
                    <div class="paper-footer">
                        <span class="date-text">${paper.pub_date}</span>
                        <div>
                            <a href="${paper.arxiv_link}" target="_blank" class="btn btn-sm btn-outline-secondary me-1">
                                <i class="bi bi-box-arrow-up-right me-1"></i>arXiv
                            </a>
                            <a href="${paper.pdf_link}" target="_blank" class="btn btn-sm btn-primary">
                                <i class="bi bi-file-earmark-pdf me-1"></i>PDF
                            </a>
                        </div>
                    </div>
                </div>
            `;
            papersList.appendChild(paperCard);
        });
    }
    
    // Apply filters
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
            
            return true;
        });
        
        renderPapers();
    }
    
    // Check if paper matches search
    function paperMatchesSearch(paper, query) {
        query = query.toLowerCase();
        return (
            paper.title.toLowerCase().includes(query) ||
            paper.authors.toLowerCase().includes(query) ||
            paper.description.toLowerCase().includes(query) ||
            (paper.abstract && paper.abstract.toLowerCase().includes(query))
        );
    }
    
    // Check if paper matches selected categories
    function paperMatchesCategories(paper, selectedCategories) {
        if (!paper.category) return false;
        
        const paperCategories = paper.category.split(',').map(cat => cat.trim());
        return selectedCategories.some(cat => paperCategories.includes(cat));
    }
    
    // Check if paper matches date range
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
    
    resetFiltersButton.addEventListener('click', function() {
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
        
        // Apply reset filters
        filteredPapers = [...allPapers];
        renderPapers();
    });
});
