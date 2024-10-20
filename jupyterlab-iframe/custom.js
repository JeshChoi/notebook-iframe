// Log that custom.js is loaded
console.log("Custom JS loaded successfully!");

// Use Jupyter's event system to ensure the notebook is fully loaded
require(["base/js/events"], function (events) {
  events.on("kernel_ready.Kernel", function () {
    console.log("Notebook initialized!"); // lets see how this does ..

    // Attach click event listener to cells
    $("#notebook-container").on("click", ".cell", function (event) {
      const cell = $(this);
      const index = $(".cell").index(cell);
      const cellContent = cell.find(".input_area").text();

      // Get the UUID from the cell's metadata, or use "N/A" if it doesn't exist
      const cellUUID = Jupyter.notebook.get_cell(index).metadata.uuid || 'N/A';

      console.log(`Cell ${index + 1} clicked:`, cellContent, `UUID: ${cellUUID}`);

      // Send a message to the parent window (React app)
      // Replace 'http://localhost:3000' with your React app's actual origin
      window.parent.postMessage(
        { action: "cellClicked", cellIndex: index, cellContent: cellContent, cellUUID: cellUUID },
        "http://localhost:4200"
      );
    });
  });
});

// Listen for messages from the parent window
window.addEventListener("message", function (event) {
  // Verify the message origin
  if (event.origin !== 'http://localhost:4200') return;

  console.log("Received message from parent:", event.data);

  if (event.data.action === "triggerCellClick") {
    // Simulate a cell click inside the Jupyter notebook
    let cellIndex = event.data.cellIndex;
    let cell = document.querySelectorAll(".cell")[cellIndex];
    if (cell) {
      cell.click();
    }
  }
}, false);
