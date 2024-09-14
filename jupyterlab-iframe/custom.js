// custom.js

// Wait until the DOM is fully loaded
window.addEventListener('DOMContentLoaded', (event) => {
    console.log('Custom JS loaded successfully');

    // Select all cells in the notebook
    const observer = new MutationObserver((mutations) => {
        mutations.forEach(() => {
            const cells = document.querySelectorAll('.cell');
            cells.forEach((cell, index) => {
                cell.addEventListener('click', () => {
                    // Send a message to the parent window with the cell index and content
                    window.parent.postMessage({ cellIndex: index, content: cell.innerText }, '*');
                });
            });
        });
    });

    observer.observe(document.body, {
        childList: true,
        subtree: true,
    });
});
