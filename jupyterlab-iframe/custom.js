// Log that custom.js is loaded
console.log("Custom JS loaded successfully!");

// Use Jupyter's event system to ensure the notebook is fully loaded
require(["base/js/events"], function (events) {
  events.on("kernel_ready.Kernel", function () {
    console.log("Notebook initialized!");

    // Attach click event listener to cells
    $("#notebook-container").on("click", ".cell", function (event) {
      const cell = $(this);
      const index = $(".cell").index(cell);
      const cellContent = cell.find(".input_area").text();

      // Get the UUID from the cell's metadata, or use "N/A" if it doesn't exist
      const cellUUID = Jupyter.notebook.get_cell(index).metadata.uuid || 'N/A';

      console.log(`Cell ${index + 1} clicked:`, cellContent, `UUID: ${cellUUID}`);

      // Send a message to the parent window (Texera app)
      window.parent.postMessage(
        { action: "cellClicked", cellIndex: index, cellContent: cellContent, cellUUID: cellUUID },
        "http://localhost:4200" // Make sure this matches your Texera app's origin
      );
    });
  });
});

// Listen for messages from the Texera app (or parent window)
window.addEventListener("message", function (event) {
  // Verify the message origin
  if (event.origin !== 'http://localhost:4200') {
    console.warn("Message received from unrecognized origin:", event.origin);
    return;
  }

  console.log("Received message in notebook:", event.data);

  if (event.data.action === "triggerCellClick") {
    const operatorCellUUIDs = event.data.operators || [];

    if (!operatorCellUUIDs.length) {
      console.error("No valid operator UUIDs provided in the message.");
      return; // Exit if no UUIDs are provided
    }

    operatorCellUUIDs.forEach((cellUUID) => {
      console.log(`Attempting to find cell with UUID: ${cellUUID}`);

      // Search for the cell by UUID
      const allCells = Jupyter.notebook.get_cells();
      const targetCell = allCells.find((cell) => cell.metadata.uuid === cellUUID);

      if (targetCell) {
        const cellIndex = Jupyter.notebook.find_cell_index(targetCell);
        console.log(`Target cell found with UUID: ${cellUUID}, at index: ${cellIndex}`);

        // Scroll to and highlight the cell
        let cell = document.querySelectorAll(".cell")[cellIndex];
        if (cell) {
          cell.scrollIntoView({ behavior: 'smooth', block: 'center' });
          cell.classList.add("highlighted");

          console.log(`Cell at index ${cellIndex} highlighted.`);

          // Remove the highlight after 3 seconds
          setTimeout(() => {
            cell.classList.remove("highlighted");
          }, 3000);
        } else {
          console.error(`Cell not found in the DOM for index ${cellIndex}.`);
        }
      } else {
        console.error(`No cell found with UUID: ${cellUUID}`);
      }
    });
  } else {
    console.warn("Received unknown action:", event.data.action);
  }
}, false);

// Add custom CSS for highlighted cells
const style = document.createElement('style');
style.innerHTML = `
  .cell.highlighted {
    background-color: lightyellow;
  }
`;
document.head.appendChild(style);
